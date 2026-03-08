/**
 * ファイルインデックス化ワークフロー
 *
 * Mastraのワークフロー機能を使った型安全なRAGパイプライン。
 * 各ステップの入出力はZodスキーマで型検証される。
 *
 * Step 1: readFilesStep  - inputs/ からファイルを読み込む
 * Step 2: chunkStep      - テキストをチャンクに分割する（MDocument）
 * Step 3: embedAndUpsertStep - Embedding生成 + ベクトルストアに保存
 */

import { createWorkflow, createStep } from "@mastra/core/workflows";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";
import { MDocument } from "@mastra/rag";
import { upsertChunks } from "../vector-store.js";

const INPUTS_DIR = "inputs";
const CHUNK_SIZE = 512;
const CHUNK_OVERLAP = 50;

// ────────────────────────────────────────────────────────────
// Step 1: ファイル読み込み
// ────────────────────────────────────────────────────────────
const readFilesStep = createStep({
  id: "read-files",
  description: "inputs/ ディレクトリからファイルを読み込む",
  inputSchema: z.object({
    inputsDir: z.string().describe("読み込み対象ディレクトリのパス"),
  }),
  outputSchema: z.object({
    files: z.array(
      z.object({
        name: z.string(),
        content: z.string(),
      })
    ),
  }),
  execute: async ({ inputData }) => {
    const dir = inputData.inputsDir;
    if (!fs.existsSync(dir)) {
      console.log(`[ingest] ${dir} が見つかりません`);
      return { files: [] };
    }

    const files = fs
      .readdirSync(dir)
      .filter((f) => {
        const stat = fs.statSync(path.join(dir, f));
        return stat.isFile() && !f.startsWith(".");
      })
      .sort()
      .map((name) => {
        const content = fs
          .readFileSync(path.join(dir, name), "utf-8")
          .replace(/\r\n/g, "\n");
        return { name, content };
      })
      .filter((f) => f.content.trim().length > 0);

    console.log(`[ingest] ${files.length} ファイルを読み込みました`);
    return { files };
  },
});

// ────────────────────────────────────────────────────────────
// Step 2: チャンク分割（MDocument）
// ────────────────────────────────────────────────────────────
const chunkStep = createStep({
  id: "chunk-documents",
  description: "MDocument でテキストをチャンクに分割する",
  inputSchema: z.object({
    files: z.array(z.object({ name: z.string(), content: z.string() })),
  }),
  outputSchema: z.object({
    chunks: z.array(
      z.object({
        text: z.string(),
        filename: z.string(),
      })
    ),
  }),
  execute: async ({ inputData }) => {
    const allChunks: Array<{ text: string; filename: string }> = [];

    for (const file of inputData.files) {
      const doc = MDocument.fromText(file.content, {
        metadata: { filename: file.name },
      });
      const chunks = await doc.chunk({
        strategy: "recursive",
        maxSize: CHUNK_SIZE,
        overlap: CHUNK_OVERLAP,
      });

      const fileChunks = chunks
        .filter((c) => c.text.trim().length > 0)
        .map((c) => ({ text: c.text, filename: file.name }));

      allChunks.push(...fileChunks);
      console.log(
        `[ingest] ${file.name}: ${fileChunks.length} チャンクに分割`
      );
    }

    return { chunks: allChunks };
  },
});

// ────────────────────────────────────────────────────────────
// Step 3: Embedding生成 + ベクトルストアへ保存
// ────────────────────────────────────────────────────────────
const embedAndUpsertStep = createStep({
  id: "embed-and-upsert",
  description: "Embedding を生成してベクトルストアに保存する",
  inputSchema: z.object({
    chunks: z.array(z.object({ text: z.string(), filename: z.string() })),
  }),
  outputSchema: z.object({
    count: z.number().describe("インデックス化したチャンク数"),
  }),
  execute: async ({ inputData }) => {
    const { chunks } = inputData;
    if (chunks.length === 0) {
      console.log("[ingest] インデックス化するチャンクがありません");
      return { count: 0 };
    }

    console.log(`[ingest] ${chunks.length} チャンクをEmbedding中...`);
    await upsertChunks(chunks);
    console.log(`[ingest] インデックス化完了: ${chunks.length} チャンク`);

    return { count: chunks.length };
  },
});

// ────────────────────────────────────────────────────────────
// ワークフロー定義
// ────────────────────────────────────────────────────────────
export const ingestWorkflow = createWorkflow({
  id: "ingest-workflow",
  description: "inputs/ のファイルをRAG用にインデックス化するパイプライン",
  inputSchema: z.object({
    inputsDir: z.string().default(INPUTS_DIR),
  }),
  outputSchema: z.object({
    count: z.number(),
  }),
  steps: [readFilesStep, chunkStep, embedAndUpsertStep],
})
  .then(readFilesStep)
  .then(chunkStep)
  .then(embedAndUpsertStep)
  .commit();
