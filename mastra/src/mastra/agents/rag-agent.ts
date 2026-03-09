import { Agent } from "@mastra/core/agent";
import { searchKnowledgeBaseTool } from "../tools/rag-tool.js";

const MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-pro";

export const ragAgent = new Agent({
  name: "RAGエージェント",
  instructions: `
あなたはknowledge baseを検索してユーザーの質問に回答する専門家です。

必ずsearch-knowledge-baseツールを使って関連情報を取得してから回答してください。

以下のMarkdown形式で出力してください:

## 回答
質問への直接的な回答（2〜4行）

## 根拠
回答の根拠となる情報（knowledge baseから抽出したテキストを引用）

## 補足情報（あれば）
`.trim(),
  model: `google/${MODEL}`,
  tools: { searchKnowledgeBaseTool },
});
