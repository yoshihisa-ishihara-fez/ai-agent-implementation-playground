/**
 * In-Memory Vector Store
 *
 * デモ用のシンプルなベクトルストア。
 * 本番環境では @mastra/pg（pgvector）などの永続化ストアを推奨。
 *
 * Embedding: paraphrase-multilingual-MiniLM-L12-v2（ローカル実行）
 */

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

interface VectorEntry {
  id: string;
  vector: number[];
  text: string;
  filename: string;
}

// インメモリストア（プロセス内で永続）
const store: VectorEntry[] = [];

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * チャンク配列をEmbeddingしてストアに追加する。
 */
export async function upsertChunks(
  chunks: Array<{ text: string; filename: string }>
): Promise<void> {
  if (chunks.length === 0) return;

  const pipe = await getEmbeddingPipeline();
  const output = await pipe(chunks.map((c) => c.text), {
    pooling: "mean",
    normalize: true,
  });
  const embeddings = output.tolist() as number[][];

  chunks.forEach((chunk, i) => {
    store.push({
      id: `${chunk.filename}-${i}`,
      vector: embeddings[i],
      text: chunk.text,
      filename: chunk.filename,
    });
  });
}

/**
 * クエリに類似したチャンクを返す。
 */
export async function searchSimilar(
  query: string,
  topK = 5
): Promise<Array<{ text: string; filename: string; score: number }>> {
  if (store.length === 0) return [];

  const pipe = await getEmbeddingPipeline();
  const output = await pipe([query], { pooling: "mean", normalize: true });
  const [embedding] = output.tolist() as number[][];

  return store
    .map((entry) => ({
      text: entry.text,
      filename: entry.filename,
      score: cosineSimilarity(embedding, entry.vector),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export function getStoreSize(): number {
  return store.length;
}
