from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """エージェントの状態を管理する構造化State"""

    messages: Annotated[list, add_messages]
    user_question: str              # ユーザーの質問
    search_results: list[dict]      # 検索結果 (ファイル名, スコア, 内容の一部)
    approved_files: list[str]       # ユーザーが承認したファイル
    file_contents: dict[str, str]   # ファイル名 → 内容
    answer: str                     # 最終回答
