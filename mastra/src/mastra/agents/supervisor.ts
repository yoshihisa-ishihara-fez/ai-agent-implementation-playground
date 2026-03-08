/**
 * スーパーバイザーエージェント
 *
 * Mastraのマルチエージェントパターン：
 * スーパーバイザーが子エージェントをツールとして呼び出す。
 * 各子エージェントは独立したAgentインスタンスとして定義されている。
 */

import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { ragAgent } from "./rag-agent.js";
import { fileAgent } from "./file-agent.js";
import { analystAgent } from "./analyst-agent.js";

const MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-pro";

// ── 子エージェント呼び出しツール ──────────────────────────────

const delegateToRagAgentTool = createTool({
  id: "delegate-to-rag-agent",
  description:
    "RAGエージェントに委譲する。knowledge baseから具体的な情報を検索・回答する場合に使う（例: 「誰が〜した？」「〜はどこ？」などの具体的な質問）。",
  inputSchema: z.object({
    question: z.string().describe("RAGエージェントへの質問"),
  }),
  outputSchema: z.object({ answer: z.string() }),
  execute: async (inputData) => {
    const result = await ragAgent.generate(inputData.question);
    return { answer: result.text };
  },
});

const delegateToFileAgentTool = createTool({
  id: "delegate-to-file-agent",
  description:
    "ファイル操作エージェントに委譲する。inputs/ のファイル一覧取得・内容読み取りが必要な場合に使う（例: 「ファイルを一覧して」「〜のファイルを読んで」）。",
  inputSchema: z.object({
    instruction: z.string().describe("ファイル操作エージェントへの指示"),
  }),
  outputSchema: z.object({ result: z.string() }),
  execute: async (inputData) => {
    const result = await fileAgent.generate(inputData.instruction);
    return { result: result.text };
  },
});

const delegateToAnalystAgentTool = createTool({
  id: "delegate-to-analyst-agent",
  description:
    "分析エージェントに委譲する。ファイル内容の要約・分析・インサイト抽出が必要な場合に使う（例: 「全ファイルを要約して」「重要ポイントを教えて」）。",
  inputSchema: z.object({
    content: z.string().describe("分析対象のテキストコンテンツ"),
  }),
  outputSchema: z.object({ analysis: z.string() }),
  execute: async (inputData) => {
    const result = await analystAgent.generate(inputData.content);
    return { analysis: result.text };
  },
});

// ── スーパーバイザー定義 ─────────────────────────────────────

export const supervisorAgent = new Agent({
  name: "スーパーバイザー",
  instructions: `
あなたはファイル分析チームのリーダーです。ユーザーの質問内容に応じて適切な専門エージェントに委譲してください。

## 委譲ルール

- **具体的な質問**（「誰が〜」「〜はどこ」「〜の受付番号は」など）
  → delegate-to-rag-agent を使う

- **ファイル操作**（「ファイル一覧を見たい」「〜のファイルを読んで」など）
  → delegate-to-file-agent を使う

- **要約・分析**（「全体を要約して」「重要ポイントを教えて」など）
  → delegate-to-file-agent でファイル内容を取得してから delegate-to-analyst-agent に渡す

複数の専門家を組み合わせることも可能です。
最終的な回答はMarkdown形式で、わかりやすくまとめてください。
`.trim(),
  model: `google/${MODEL}`,
  tools: {
    delegateToRagAgentTool,
    delegateToFileAgentTool,
    delegateToAnalystAgentTool,
  },
});
