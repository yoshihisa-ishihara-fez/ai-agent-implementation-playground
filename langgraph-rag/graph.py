from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import (
    index_node,
    question_node,
    search_node,
    review_node,
    read_node,
    answer_node,
    error_node,
)
from state import AgentState


def _route_after_search(state: AgentState) -> str:
    """search_node 後の条件分岐: 検索結果があるか？"""
    if state.get("search_results"):
        return "review"
    return "error"


def _route_after_review(state: AgentState) -> str:
    """review_node 後の条件分岐: 承認されたか？"""
    if state.get("approved_files"):
        return "read"
    return "question"  # 拒否 → 質問やり直し


def build_graph():
    """RAGエージェントのグラフを構築"""
    builder = StateGraph(AgentState)

    # ── ノード登録 ──
    builder.add_node("index", index_node)
    builder.add_node("question", question_node)
    builder.add_node("search", search_node)
    builder.add_node("review", review_node)
    builder.add_node("read", read_node)
    builder.add_node("answer", answer_node)
    builder.add_node("error", error_node)

    # ── エッジ定義 ──
    builder.add_edge(START, "index")
    builder.add_edge("index", "question")
    builder.add_edge("question", "search")

    # 検索結果の有無で分岐
    builder.add_conditional_edges("search", _route_after_search, {
        "review": "review",
        "error": "error",
    })

    # ユーザー承認の有無で分岐（拒否時はquestionに戻る）
    builder.add_conditional_edges("review", _route_after_review, {
        "read": "read",
        "question": "question",
    })

    builder.add_edge("read", "answer")
    builder.add_edge("answer", END)
    builder.add_edge("error", END)

    # ── コンパイル（チェックポインタ付き: interrupt に必要） ──
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
