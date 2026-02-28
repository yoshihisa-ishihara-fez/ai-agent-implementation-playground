from pathlib import Path

from agno.tools import tool

INPUTS_DIR = Path("inputs")


@tool
def list_files() -> str:
    """inputs/ ディレクトリ内のファイル一覧を返す。ファイルが存在しない場合はその旨を返す。"""
    if not INPUTS_DIR.exists():
        return "inputs/ ディレクトリが存在しません。"

    files = sorted([
        f.name for f in INPUTS_DIR.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ])

    if not files:
        return "inputs/ にファイルがありません。"

    return "\n".join(files)


@tool
def read_file(filename: str) -> str:
    """inputs/ ディレクトリ内の指定ファイルの内容を返す。

    Args:
        filename: ファイル名（パス区切り文字なし、例: sample1.txt）
    """
    name = filename.strip()

    # バリデーション: パストラバーサル防止
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return f"エラー: 不正なファイル名です: {filename!r}"

    p = INPUTS_DIR / name

    try:
        resolved = p.resolve()
        if not str(resolved).startswith(str(INPUTS_DIR.resolve())):
            return f"エラー: inputs/ 外のファイルは読めません: {filename}"
    except Exception as e:
        return f"エラー: パス解決に失敗しました: {e}"

    if not p.exists():
        return f"エラー: ファイルが見つかりません: {filename}"
    if not p.is_file():
        return f"エラー: ディレクトリは読めません: {filename}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")[:8000]
        return content if content.strip() else "(空ファイル)"
    except Exception as e:
        return f"エラー: 読み取りに失敗しました: {e}"
