import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { searchSimilar } from "../vector-store.js";

export const searchKnowledgeBaseTool = createTool({
  id: "search-knowledge-base",
  description:
    "インデックス化されたファイルの知識ベースをベクトル検索（RAG）で照会し、質問に関連する情報を返す。具体的な質問への回答に使う。",
  inputSchema: z.object({
    query: z.string().describe("知識ベースに対する検索クエリ"),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        text: z.string(),
        filename: z.string(),
        score: z.number(),
      })
    ),
  }),
  execute: async ({ context }) => {
    const results = await searchSimilar(context.query, 5);
    return { results };
  },
});
