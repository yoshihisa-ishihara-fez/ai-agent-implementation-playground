import os
import subprocess
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


INPUTS_DIR = Path("inputs")

@tool
def ls_inputs() -> str:
    """
    inputs/ のファイル一覧を表示します（固定コマンド: ls -la inputs）。
    破壊的操作は行いません。
    """
    if not INPUTS_DIR.exists():
        return "inputs/ が存在しません。"

    cp = subprocess.run(
        ["ls", "-la", str(INPUTS_DIR)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    out = (cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")
    return out[-8000:]


def extract_text(content) -> str:
    """
    LLM の応答 content を整形済みテキストに変換する。
    - str の場合: そのまま返す
    - list (content blocks) の場合: type=="text" のブロックだけ結合
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and "text" in block
        ]
        return "\n".join(text_parts)

    return str(content)

@tool
def read_file(filename: str) -> str:
    """
    inputs/ 直下の指定ファイルを読み取り、内容をテキストで返します。
    filename はファイル名のみ（パス区切り不可、隠しファイル不可）。
    最大 8000 文字まで返します。
    """
    # バリデーション: パストラバーサル防止
    name = filename.strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return f"エラー: 不正なファイル名です: {filename!r}"

    p = INPUTS_DIR / name

    # シンボリックリンクによる脱出も防止
    try:
        resolved = p.resolve()
        if not str(resolved).startswith(str(INPUTS_DIR.resolve())):
            return f"エラー: inputs/ 外のファイルは読めません: {filename}"
    except Exception:
        return f"エラー: パス解決に失敗しました: {filename}"

    if not p.exists():
        return f"エラー: ファイルが見つかりません: {filename}"
    if not p.is_file():
        return f"エラー: ディレクトリは読めません: {filename}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")[:8000]
        return content if content else "(空ファイル)"
    except Exception as e:
        return f"エラー: 読み取りに失敗しました: {e}"


def main() -> None:
    # 0) Setup
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY が未設定です。.env を確認してください。")

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model_name,
        temperature=0.2,
    )

    # 1) Agent: ls_inputs と read_file の両方をツールとして渡す
    #    エージェントが自律的に「一覧取得 → ファイル選択 → 読み取り → 要約」を行う
    agent = create_agent(
        model=llm,
        tools=[ls_inputs, read_file],
        system_prompt=(
            "あなたはファイル要約エージェントです。以下の手順で作業してください。\n"
            "\n"
            "1. まず ls_inputs を呼び出して inputs/ のファイル一覧を取得する\n"
            "2. 一覧から重要そうなファイルを最大3つ選ぶ（ディレクトリは除外）\n"
            "3. 選んだファイルを read_file で1つずつ読み取る\n"
            "4. 読み取った内容をもとに、以下の形式で Markdown 要約を出力する:\n"
            "\n"
            "## サマリ（3行以内）\n"
            "## 重要ポイント（3〜7個）\n"
            "## 不明点/追加で欲しい情報（あれば）\n"
            "## 次アクション（最大3つ）\n"
            "\n"
            "すべてのツール呼び出しが完了してから最終的な要約を出力してください。"
        ),
    )

    # 2) Run agent
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "inputs/ を調査して、重要そうなファイルを読み取り、要約してください。",
            }
        ]
    })

    # 3) 最終メッセージ（AI の要約）を出力
    last_msg = result["messages"][-1]
    content = getattr(last_msg, "content", str(last_msg))
    print(extract_text(content))


if __name__ == "__main__":
    main()