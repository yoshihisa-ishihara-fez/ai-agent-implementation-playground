"""StateGraph の構築・コンパイル."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import (
    error_node,
    human_review_node,
    ls_node,
    read_node,
    select_node,
    summarize_node,
)
from state import AgentState


def _route_after_select(state: AgentState) -> str:
    """select_node 後の条件分岐: ファイルが選択されたか？"""
    if state.get("selected_files"):
        return "human_review"
    return "error"


def _route_after_review(state: AgentState) -> str:
    """human_review_node 後の条件分岐: 承認されたか？"""
    if state.get("approved_files"):
        return "read"
    # 拒否 → select_node に戻って再選択
    return "select"


def build_graph():
    """StateGraph を構築・コンパイルして返す。

    グラフ構造:
        START → ls → select → [ファイルあり?]
                                 ├─ No  → error → END
                                 └─ Yes → human_review → [承認?]
                                                          ├─ No  → select（ループ）
                                                          └─ Yes → read → summarize → END
    """
    builder = StateGraph(AgentState)

    # ── ノード登録 ──
    builder.add_node("ls", ls_node)
    builder.add_node("select", select_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("read", read_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("error", error_node)

    # ── エッジ定義 ──
    builder.add_edge(START, "ls")
    builder.add_edge("ls", "select")

    # select → 条件分岐
    builder.add_conditional_edges("select", _route_after_select, {
        "human_review": "human_review",
        "error": "error",
    })

    # human_review → 条件分岐
    builder.add_conditional_edges("human_review", _route_after_review, {
        "read": "read",
        "select": "select",  # 拒否時のループ
    })

    # read → summarize → END
    builder.add_edge("read", "summarize")
    builder.add_edge("summarize", END)
    builder.add_edge("error", END)

    # ── コンパイル（チェックポインタ付き: interrupt に必要） ──
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
