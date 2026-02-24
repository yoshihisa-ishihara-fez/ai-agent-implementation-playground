import os

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt

from state import AgentState
from tools import read_file
from vectorstore import index_documents, search_documents, init_vectorstore


def _get_llm() -> ChatGoogleGenerativeAI:
    """LLM インスタンスを生成する。"""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
    )


def index_node(state: AgentState) -> dict:
    """inputs/ のファイルをベクトルストアにインデックス化する。"""
    print("[index_node] ファイルのインデックス化を開始...")
    vectorstore = index_documents()
    print("[index_node] インデックス化完了")
    return {}


def question_node(state: AgentState) -> dict:
    """ユーザーからの質問を受け取る（interrupt で待機）。"""
    question = interrupt({
        "prompt": "何について知りたいですか？質問を入力してください:",
        "instruction": "質問内容を入力してください。",
    })

    print(f"[question_node] ユーザーの質問: {question}")
    return {
        "user_question": question,
        "messages": [HumanMessage(content=question)],
    }


def search_node(state: AgentState) -> dict:
    """ユーザーの質問をもとにベクトル検索を実行する。"""
    question = state.get("user_question", "")
    if not question:
        return {"search_results": []}

    print(f"[search_node] 検索中: '{question}'")
    vectorstore = init_vectorstore()
    results = search_documents(query=question, k=3, vectorstore=vectorstore)

    print(f"[search_node] {len(results)} 件の検索結果を取得")
    for r in results:
        print(f"  - {r['filename']} (スコア: {r['score']:.4f})")

    return {"search_results": results}


def review_node(state: AgentState) -> dict:
    """検索結果をユーザーに提示し、承認を待つ（interrupt）。"""
    results = state.get("search_results", [])

    filenames = [r["filename"] for r in results]

    decision = interrupt({
        "question": "以下のファイルから情報を取得してよいですか？",
        "search_results": results,
        "instruction": "承認する場合は 'yes'、質問をやり直す場合は 'no' と入力してください。",
    })

    if decision == "yes":
        print(f"[review_node] ユーザー承認: {filenames}")
        return {"approved_files": filenames}
    else:
        print("[review_node] ユーザー拒否 → 質問をやり直します")
        return {"approved_files": []}


def read_node(state: AgentState) -> dict:
    """承認されたファイルを読み取る。"""
    approved = state.get("approved_files", [])
    contents: dict[str, str] = {}

    for filename in approved:
        content = read_file(filename)
        contents[filename] = content
        print(f"[read_node] 読み取り完了: {filename} ({len(content)} 文字)")

    return {"file_contents": contents}


def answer_node(state: AgentState) -> dict:
    """ユーザーの質問に対して、読み取ったファイル内容をもとに回答を生成する。"""
    question = state.get("user_question", "")
    contents = state.get("file_contents", {})

    files_text = "\n\n".join(
        f"### {name}\n```\n{text}\n```"
        for name, text in contents.items()
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "あなたは質問応答エージェントです。\n"
            "与えられたファイル内容をもとに、ユーザーの質問に的確に回答してください。\n"
            "回答は以下の形式で Markdown で出力してください:\n"
            "\n"
            "## 回答\n"
            "質問に対する直接的な回答（2〜4行）\n"
            "\n"
            "## 根拠\n"
            "回答の根拠となる情報（箇条書き）\n"
            "\n"
            "## 補足情報（あれば）\n"
            "追加で役立つ情報や関連事項\n"
        )),
        HumanMessage(content=f"質問: {question}\n\n参照ファイル:\n{files_text}"),
    ])

    answer = response.content
    print("[answer_node] 回答生成完了")

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def error_node(state: AgentState) -> dict:
    """検索結果が見つからなかった場合のエラー処理。"""
    question = state.get("user_question", "")
    msg = f"'{question}' に関連するファイルが見つかりませんでした。別の質問を試してください。"
    print(f"[error_node] {msg}")
    return {
        "answer": msg,
        "messages": [AIMessage(content=msg)],
    }
