import os

from agno.agent import Agent
from agno.models.google import Gemini
from agno.team import Team

from knowledge import build_knowledge
from tools import list_files, read_file


def _get_model() -> Gemini:
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return Gemini(id=model_name)


def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY が未設定です。.env を確認してください。")

    print("=" * 60)
    print("Agno マルチエージェント ファイル分析システム（RAG対応）")
    print("=" * 60)

    # ── Knowledge base 構築（RAG用インデックス化）──────────────
    print("\nファイルをインデックス化中...")
    knowledge = build_knowledge()
    print()

    # ── RAGエージェント ───────────────────────────────────────
    # knowledge base を検索してユーザーの質問に回答する
    rag_agent = Agent(
        name="RAGエージェント",
        role="knowledge base を検索してユーザーの質問に的確に回答する専門家",
        model=_get_model(),
        knowledge=knowledge,
        search_knowledge=True,
        instructions=[
            "knowledge base から関連情報を検索し、ユーザーの質問に回答します。",
            "以下のMarkdown形式で出力してください:",
            "## 回答",
            "質問への直接的な回答（2〜4行）",
            "## 根拠",
            "回答の根拠となる情報（knowledge base から抽出）",
            "## 補足情報（あれば）",
        ],
        markdown=True,
    )

    # ── ファイル操作エージェント ──────────────────────────────
    # inputs/ のファイルを一覧表示・読み取りする
    file_agent = Agent(
        name="ファイル操作エージェント",
        role="inputs/ ディレクトリのファイルを一覧表示・読み取りする専門家",
        model=_get_model(),
        tools=[list_files, read_file],
        instructions=[
            "inputs/ ディレクトリ内のファイル操作を担当します。",
            "まず list_files でファイル一覧を取得し、重要そうなファイルを最大3つ選んでください。",
            "選んだファイルを read_file で1つずつ読み取り、内容をそのまま報告してください。",
        ],
    )

    # ── 分析エージェント ──────────────────────────────────────
    # ファイル内容を受け取り、分析・要約を担当
    analyst_agent = Agent(
        name="分析エージェント",
        role="ファイル内容を分析・要約・インサイト抽出する専門家",
        model=_get_model(),
        instructions=[
            "ファイル操作エージェントから受け取った内容を分析・要約します。",
            "以下のMarkdown形式で出力してください:",
            "## サマリ（3行以内）",
            "## 重要ポイント（箇条書き 3〜7個）",
            "## 不明点/追加で欲しい情報（あれば）",
            "## 次アクション（最大3つ）",
        ],
        markdown=True,
    )

    # ── チーム定義 ────────────────────────────────────────────
    # coordinate モード: リーダーがタスクを分解し各エージェントに委譲
    team = Team(
        name="ファイル分析チーム",
        mode="coordinate",
        model=_get_model(),
        members=[rag_agent, file_agent, analyst_agent],
        instructions=[
            "ファイル分析チームのリーダーとして振る舞ってください。",
            "ユーザーが具体的な質問をした場合は RAGエージェント に委譲してください。",
            "一般的な要約を求められた場合はファイル操作エージェントと分析エージェントを使ってください。",
        ],
        markdown=True,
        show_members_responses=True,
    )

    # ── 対話ループ ────────────────────────────────────────────
    print("質問を入力してください（'quit' で終了）:")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n>> ").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("終了します。")
            break

        print()
        team.print_response(
            input=user_input,
            stream=True,
        )


if __name__ == "__main__":
    main()
