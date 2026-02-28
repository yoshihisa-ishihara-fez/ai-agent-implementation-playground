from pathlib import Path

from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb

INPUTS_DIR = Path("inputs")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "file_contents"

# langgraph-rag の時と同じ多言語対応モデルを使用
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_knowledge() -> Knowledge:
    """inputs/ のファイルをインデックス化した Knowledge base を構築して返す。"""
    embedder = SentenceTransformerEmbedder(
        id=EMBEDDING_MODEL,
    )

    knowledge = Knowledge(
        vector_db=ChromaDb(
            collection=COLLECTION_NAME,
            path=CHROMA_DIR,
            persistent_client=True,
            embedder=embedder,
        )
    )

    files = sorted([
        f for f in INPUTS_DIR.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ])

    if not files:
        print("[knowledge] inputs/ にファイルがありません")
        return knowledge

    for file in files:
        try:
            content = file.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                knowledge.insert(
                    name=file.name,
                    text_content=content,
                    metadata={"filename": file.name},
                )
                print(f"[knowledge] インデックス化完了: {file.name}")
        except Exception as e:
            print(f"[knowledge] {file.name} の処理に失敗: {e}")

    return knowledge
