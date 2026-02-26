import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any


# -------- LLM client (Gemini only) --------

def chat_complete(messages: List[Dict[str, str]]) -> str:
    # NOTE: SDKの取り出し方が環境で違う可能性はあります。差し替え点はここだけ。
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    sys_text = "\n".join([m["content"] for m in messages if m["role"] == "system"])
    user_text = "\n".join([m["content"] for m in messages if m["role"] == "user"])
    merged = f"【System】\n{sys_text}\n\n【User】\n{user_text}".strip()

    resp = client.models.generate_content(model=model, contents=merged)
    return getattr(resp, "text", "") or ""


# -------- Tool: fixed command (one command only) --------

def run_fixed_ls(inputs_dir: Path) -> str:
    # 固定コマンドはこれだけ
    cmd = ["ls", "-la", str(inputs_dir)]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    return out[-8000:]


# -------- File reader (Python, not a tool command) --------

def read_file_safe(path: Path, max_chars: int = 8000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except Exception as e:
        return f"<<failed to read {path.name}: {e}>>"


# -------- Agent prompts --------

SELECT_SYSTEM = """\
あなたは「重要ファイル抽出エージェント」です。
inputs/ のファイル一覧（ls結果）を見て、重要そうなファイルを最大3つ選んでください。

制約:
- 出力はJSONのみ
- 選ぶのは inputs/ 直下のファイル名（ディレクトリは選ばない）
- 根拠も短く添える

出力フォーマット:
{
  "selected": [
    {"name": "sample1.txt", "reason": "..." },
    {"name": "notes.md", "reason": "..." }
  ]
}
"""

SUMMARIZE_SYSTEM = """\
あなたは「要約エージェント」です。
与えられたファイル内容を、実務で使える形に要約してください。

出力はMarkdownで:
- サマリ（3行以内）
- 重要ポイント（箇条書き 3〜7）
- 不明点/追加で欲しい情報（あれば）
- 次アクション（最大3つ）
"""


def main() -> None:
    inputs_dir = Path("inputs")

    if not inputs_dir.exists():
        print("inputs/ がありません。inputs/ を作ってファイルを入れてください。")
        return

    # 1) Observe (fixed command)
    ls_output = run_fixed_ls(inputs_dir)

    # 2) Decide: pick important files
    select_text = chat_complete([
        {"role": "system", "content": SELECT_SYSTEM},
        {"role": "user", "content": ls_output},
    ])

    try:
        selected_json = json.loads(select_text)
        selected = selected_json.get("selected", [])
    except Exception:
        # LLMがJSON崩したときの最低限フォールバック: txt/md を上位から最大3件
        selected = []
        for line in ls_output.splitlines():
            parts = line.split()
            if not parts:
                continue
            name = parts[-1]
            if name.endswith((".txt", ".md")):
                selected.append({"name": name, "reason": "fallback: listed by ls"})
        selected = selected[:3]

    # 3) Read selected files (no extra command)
    payload: Dict[str, Any] = {"ls": ls_output, "files": []}

    for item in selected[:3]:
        name = (item.get("name") or "").strip()
        if not name or "/" in name or name.startswith("."):
            continue
        p = inputs_dir / name
        if p.exists() and p.is_file():
            payload["files"].append({
                "name": name,
                "reason": item.get("reason", ""),
                "content": read_file_safe(p),
            })

    if not payload["files"]:
        print("# No files selected\n")
        print("LLMがファイルを選べませんでした。inputs/ に txt/md を入れて再実行してください。")
        print("\n## ls output\n")
        print(ls_output)
        print("\n## raw LLM output\n")
        print(select_text)
        return

    # 4) Summarize
    report = chat_complete([
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ])

    print(report)


if __name__ == "__main__":
    main()
