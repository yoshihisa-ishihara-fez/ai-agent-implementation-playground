/**
 * エントリーポイント
 *
 * 1. インデックス化ワークフローを実行（RAGパイプライン）
 * 2. スーパーバイザーエージェントと対話ループ
 */

import "dotenv/config";
import * as readline from "readline";
import { mastra } from "./mastra/index.js";
import { supervisorAgent } from "./mastra/agents/supervisor.js";

async function runIngestWorkflow(): Promise<void> {
  const workflow = mastra.getWorkflow("ingestWorkflow");
  const run = workflow.createRun();
  await run.start({ inputData: { inputsDir: "inputs" } });
}

async function main(): Promise<void> {
  if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
    throw new Error(
      "GOOGLE_GENERATIVE_AI_API_KEY が未設定です。.env を確認してください。"
    );
  }

  console.log("=".repeat(60));
  console.log("Mastra マルチエージェント ファイル分析システム（RAG対応）");
  console.log("=".repeat(60));

  // ── RAGパイプライン（インデックス化ワークフロー）実行 ──
  console.log("\nファイルをインデックス化中...");
  await runIngestWorkflow();
  console.log();

  // ── 対話ループ ────────────────────────────────────────────
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("質問を入力してください（'quit' で終了）:");
  console.log("-".repeat(60));

  const ask = (): void => {
    rl.question("\n>> ", async (input) => {
      const trimmed = input.trim();

      if (!trimmed) {
        ask();
        return;
      }

      if (["quit", "exit", "q"].includes(trimmed.toLowerCase())) {
        console.log("終了します。");
        rl.close();
        return;
      }

      try {
        console.log();
        const response = await supervisorAgent.stream(trimmed);
        for await (const chunk of response.textStream) {
          process.stdout.write(chunk);
        }
        console.log("\n");
      } catch (e) {
        console.error("エラー:", e);
      }

      ask();
    });
  };

  ask();
}

main().catch(console.error);
