"""エントリポイント — グラフの実行と Human-in-the-loop のインタラクション."""

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
    # 0) 環境チェック
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY が未設定です。.env を確認してください。")

    # 1) グラフをビルド
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("=" * 60)
    print("LangGraph ファイル要約エージェント")
    print("=" * 60)

    # 2) 初回実行 — ls → select まで進み、human_review で interrupt される
    initial_input = {
        "messages": [{"role": "user", "content": "inputs/ を調査して要約してください。"}],
        "file_list": [],
        "selected_files": [],
        "approved_files": [],
        "file_contents": {},
        "summary": "",
    }

    result = graph.invoke(initial_input, config=config)

    # 3) Human-in-the-loop ループ
    while True:
        # interrupt が発生したかチェック
        state = graph.get_state(config)

        if not state.tasks:
            # interrupt なし → グラフ完了
            break

        # interrupt の内容を表示
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    info = intr.value
                    print()
                    print("-" * 40)
                    print(f"  {info.get('question', '')}")
                    for f in info.get("selected_files", []):
                        print(f"    - {f}")
                    print(f"  {info.get('instruction', '')}")
                    print("-" * 40)

        # ユーザー入力を受け取る
        user_input = input("\n>> ").strip().lower()
        decision = "yes" if user_input in ("yes", "y", "") else "no"

        # resume で再開
        result = graph.invoke(Command(resume=decision), config=config)

    # 4) 最終結果を出力
    final_state = graph.get_state(config)
    summary = final_state.values.get("summary", "")

    print()
    print("=" * 60)
    print("最終要約")
    print("=" * 60)
    print(extract_text(summary))


if __name__ == "__main__":
    main()
