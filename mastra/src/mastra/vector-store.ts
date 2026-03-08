/**
 * In-Memory Vector Store
 *
 * デモ用のシンプルなベクトルストア。
 * 本番環境では @mastra/pg（pgvector）などの永続化ストアを推奨。
 */

import { embedMany, embed } from "ai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";

// text-embedding-004 は v1 API でのみ動作するため baseURL を明示的に指定
const googleV1 = createGoogleGenerativeAI({
  baseURL: "https://generativelanguage.googleapis.com/v1",
});
const EMBEDDING_MODEL = googleV1.textEmbeddingModel("text-embedding-004");

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

  const { embeddings } = await embedMany({
    model: EMBEDDING_MODEL,
    values: chunks.map((c) => c.text),
  });

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

  const { embedding } = await embed({
    model: EMBEDDING_MODEL,
    value: query,
  });

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
