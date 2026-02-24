import os
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from tools import list_files, INPUTS_DIR

CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "file_contents"


def get_embeddings() -> HuggingFaceEmbeddings:
    """HuggingFace Embeddings を取得（無料、APIキー不要）"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def init_vectorstore() -> Chroma:
    """ベクトルストアを初期化（既存があれば読み込み、なければ作成）"""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    return vectorstore


def index_documents(vectorstore: Optional[Chroma] = None) -> Chroma:
    """inputs/ 内のファイルをベクトルストアにインデックス化"""
    if vectorstore is None:
        vectorstore = init_vectorstore()

    files = list_files()
    if not files:
        print("[vectorstore] inputs/ にファイルがありません。")
        return vectorstore

    documents = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={
                        "filename": file_path.name,
                        "source": str(file_path),
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"[vectorstore] {file_path.name} の読み取りに失敗: {e}")

    if documents:
        # 既存のコレクションをクリアして再インデックス
        vectorstore.reset_collection()
        vectorstore.add_documents(documents)
        print(f"[vectorstore] {len(documents)} 件のドキュメントをインデックス化しました。")
    else:
        print("[vectorstore] インデックス可能なドキュメントがありませんでした。")

    return vectorstore


def search_documents(query: str, k: int = 3, vectorstore: Optional[Chroma] = None) -> list[dict]:
    """
    ベクトル検索で類似ドキュメントを取得

    Returns:
        list[dict]: [{"filename": str, "score": float, "preview": str}, ...]
    """
    if vectorstore is None:
        vectorstore = init_vectorstore()

    results = vectorstore.similarity_search_with_score(query, k=k)

    search_results = []
    for doc, score in results:
        filename = doc.metadata.get("filename", "unknown")
        preview = doc.page_content[:100].replace("\n", " ")
        search_results.append({
            "filename": filename,
            "score": float(score),
            "preview": preview,
        })

    return search_results
