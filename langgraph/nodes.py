"""ノード関数群 — StateGraph の各ノードで実行される処理."""

import json
import os

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt

from state import AgentState
from tools import list_filenames, read_file


def _get_llm() -> ChatGoogleGenerativeAI:
    """LLM インスタンスを生成する。"""
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
    )


# ── ノード関数 ──────────────────────────────────────────


def ls_node(state: AgentState) -> dict:
    """inputs/ のファイル一覧を取得し file_list に格納する。"""
    files = list_filenames()
    print(f"[ls_node] 検出ファイル: {files}")
    return {"file_list": files}


def select_node(state: AgentState) -> dict:
    """LLM にファイル一覧を渡し、重要なファイルを最大3つ選ばせる。"""
    file_list = state.get("file_list", [])

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "あなたはファイル選択アシスタントです。\n"
            "与えられたファイル一覧から、重要そうなファイルを最大3つ選んでください。\n"
            "回答は JSON 配列のみで返してください（例: [\"file1.txt\", \"file2.md\"]）。\n"
            "余計な説明は不要です。"
        )),
        HumanMessage(content=f"ファイル一覧:\n{json.dumps(file_list, ensure_ascii=False)}"),
    ])

    # LLM の応答から JSON 配列をパース
    try:
        text = response.content.strip()
        # ```json ... ``` で囲まれている場合の対応
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        selected = json.loads(text)
        if not isinstance(selected, list):
            selected = file_list[:3]
    except (json.JSONDecodeError, IndexError):
        selected = file_list[:3]

    # file_list に存在するものだけに絞る
    selected = [f for f in selected if f in file_list][:3]

    print(f"[select_node] LLM が選択: {selected}")
    return {
        "selected_files": selected,
        "messages": [AIMessage(content=f"以下のファイルを選択しました: {selected}")],
    }


def human_review_node(state: AgentState) -> dict:
    """interrupt() でユーザーに選択内容を提示し、承認を待つ。"""
    selected = state.get("selected_files", [])

    # ここで実行が一時停止し、ユーザーの応答を待つ
    decision = interrupt({
        "question": "以下のファイルを読み取ってよいですか？",
        "selected_files": selected,
        "instruction": "承認する場合は 'yes'、やり直す場合は 'no' と入力してください。",
    })

    if decision == "yes":
        print(f"[human_review] ユーザー承認: {selected}")
        return {"approved_files": selected}
    else:
        print(f"[human_review] ユーザー拒否 → 再選択へ")
        return {"approved_files": []}


def read_node(state: AgentState) -> dict:
    """approved_files のファイルを順に読み取り file_contents に格納する。"""
    approved = state.get("approved_files", [])
    contents: dict[str, str] = {}

    for filename in approved:
        content = read_file(filename)
        contents[filename] = content
        print(f"[read_node] 読み取り完了: {filename} ({len(content)} 文字)")

    return {"file_contents": contents}


def summarize_node(state: AgentState) -> dict:
    """LLM に file_contents を渡して構造化された要約を生成する。"""
    contents = state.get("file_contents", {})

    # ファイル内容を整形
    files_text = "\n\n".join(
        f"### {name}\n```\n{text}\n```"
        for name, text in contents.items()
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "あなたはファイル要約エージェントです。\n"
            "与えられたファイル内容をもとに、以下の形式で Markdown 要約を出力してください:\n"
            "\n"
            "## サマリ（3行以内）\n"
            "## 重要ポイント（3〜7個）\n"
            "## 不明点/追加で欲しい情報（あれば）\n"
            "## 次アクション（最大3つ）\n"
        )),
        HumanMessage(content=f"以下のファイルを要約してください:\n\n{files_text}"),
    ])

    summary = response.content
    print("[summarize_node] 要約生成完了")
    return {
        "summary": summary,
        "messages": [AIMessage(content=summary)],
    }


def error_node(state: AgentState) -> dict:
    """ファイルが見つからなかった場合のエラー処理。"""
    msg = "対象ファイルが見つかりませんでした。inputs/ にファイルを配置してください。"
    print(f"[error_node] {msg}")
    return {
        "summary": msg,
        "messages": [AIMessage(content=msg)],
    }
