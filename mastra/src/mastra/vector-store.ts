/**
 * ChromaDB + HuggingFace Embedding
 *
 * ChromaDB HTTP クライアント経由でベクトル検索を行う。
 * Embedding: paraphrase-multilingual-MiniLM-L12-v2（ローカル実行）
 */

import { ChromaClient } from "chromadb";
import { pipeline, env } from "@huggingface/transformers";

env.cacheDir = process.env.HF_CACHE_DIR ?? ".cache/transformers";

const MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

type EmbeddingPipeline = Awaited<ReturnType<typeof pipeline<"feature-extraction">>>;
let _pipe: EmbeddingPipeline | null = null;

async function getEmbeddingPipeline(): Promise<EmbeddingPipeline> {
  if (!_pipe) {
    _pipe = await pipeline("feature-extraction", MODEL_ID);
  }
  return _pipe;
}

// HuggingFace モデルを ChromaDB のカスタム Embedding 関数として渡す
const embeddingFunction = {
  generate: async (texts: string[]): Promise<number[][]> => {
    const pipe = await getEmbeddingPipeline();
    const output = await pipe(texts, { pooling: "mean", normalize: true });
    return output.tolist() as number[][];
  },
};

const client = new ChromaClient({
  path: `http://${process.env.CHROMA_HOST ?? "localhost"}:${process.env.CHROMA_PORT ?? "8000"}`,
});

const collection = await client.getOrCreateCollection({
  name: "file_contents",
  embeddingFunction,
  metadata: { "hnsw:space": "cosine" },
});

/**
 * チャンク配列をEmbeddingしてChromaDBに追加する。
 */
export async function upsertChunks(
  chunks: Array<{ text: string; filename: string }>
): Promise<void> {
  if (chunks.length === 0) return;

  await collection.upsert({
    ids: chunks.map((c, i) => `${c.filename}-${i}`),
    documents: chunks.map((c) => c.text),
    metadatas: chunks.map((c) => ({ filename: c.filename })),
  });
}

/**
 * クエリに類似したチャンクを返す。
 */
export async function searchSimilar(
  query: string,
  topK = 5
): Promise<Array<{ text: string; filename: string; score: number }>> {
  const results = await collection.query({ queryTexts: [query], nResults: topK });
  return results.documents[0].map((doc, i) => ({
    text: doc ?? "",
    filename: (results.metadatas[0][i] as { filename: string }).filename,
    score: 1 - (results.distances?.[0][i] ?? 0), // cosine: distance = 1 - similarity
  }));
}

export async function getStoreSize(): Promise<number> {
  return collection.count();
}
