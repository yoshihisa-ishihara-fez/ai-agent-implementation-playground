import os
import uuid

from langgraph.types import Command
from graph import build_graph


def extract_text(content) -> str:
    """LLM の応答 content をテキストに変換する。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def main() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY が未設定です。.env を確認してください。")

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("=" * 60)
    print("LangGraph + RAG ファイル検索エージェント")
    print("=" * 60)

    # 初期状態
    initial_input = {
        "messages": [],
        "user_question": "",
        "search_results": [],
        "approved_files": [],
        "file_contents": {},
        "answer": "",
    }

    # グラフ実行開始
    result = graph.invoke(initial_input, config=config)

    # Human-in-the-loop ループ
    while True:
        state = graph.get_state(config)

        # タスクがなければ終了
        if not state.tasks:
            break

        # interrupt の内容を表示
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    info = intr.value

                    print()
                    print("-" * 60)

                    # question_node の interrupt
                    if "prompt" in info:
                        print(f"  {info.get('prompt', '')}")
                        print(f"  {info.get('instruction', '')}")

                    # review_node の interrupt
                    elif "question" in info:
                        print(f"  {info.get('question', '')}")
                        search_results = info.get("search_results", [])
                        for r in search_results:
                            score = r.get("score", 0)
                            preview = r.get("preview", "")
                            print(f"    - {r['filename']} (スコア: {score:.4f})")
                            print(f"      プレビュー: {preview}...")
                        print(f"  {info.get('instruction', '')}")

                    print("-" * 60)

        # ユーザー入力
        user_input = input("\n>> ").strip()

        # review_node の場合は yes/no 判定
        # question_node の場合はそのまま質問として渡す
        state_values = graph.get_state(config).values
        if state_values.get("search_results"):
            # review_node の場合
            decision = "yes" if user_input.lower() in ("yes", "y", "") else "no"
            result = graph.invoke(Command(resume=decision), config=config)
        else:
            # question_node の場合
            result = graph.invoke(Command(resume=user_input), config=config)

    # 最終結果を出力
    final_state = graph.get_state(config)
    answer = final_state.values.get("answer", "")

    print()
    print("=" * 60)
    print("最終回答")
    print("=" * 60)
    print(extract_text(answer))


if __name__ == "__main__":
    main()
