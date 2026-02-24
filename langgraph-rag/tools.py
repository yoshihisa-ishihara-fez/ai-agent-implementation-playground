from pathlib import Path

INPUTS_DIR = Path("inputs")


def read_file(filename: str) -> str:
    """
    inputs/ 直下の指定ファイルを読み取り、内容をテキストで返す。
    パストラバーサル防止・シンボリックリンク脱出防止付き。最大 8000 文字まで。
    """
    name = filename.strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return f"エラー: 不正なファイル名です: {filename!r}"

    p = INPUTS_DIR / name

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


def list_files() -> list[Path]:
    """inputs/ 内のファイルリストを返す（ディレクトリ・隠しファイル除外）。"""
    if not INPUTS_DIR.exists():
        return []
    return sorted(
        p for p in INPUTS_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
