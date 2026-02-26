"""AgentState — グラフ全体で共有される状態の定義."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """各ノードが読み書きする共有状態。

    LangChain 版では messages だけに全情報が埋もれていたが、
    LangGraph 版ではステップごとの中間結果を個別フィールドで管理する。
    """

    messages: Annotated[list, add_messages]
    file_list: list[str]  # ls_node で取得したファイル名一覧
    selected_files: list[str]  # LLM が選んだファイル（最大3つ）
    approved_files: list[str]  # ユーザーが承認したファイル
    file_contents: dict[str, str]  # ファイル名 → 内容
    summary: str  # 最終要約
