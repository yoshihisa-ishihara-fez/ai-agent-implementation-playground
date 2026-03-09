import { Agent } from "@mastra/core/agent";

const MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-pro";

export const analystAgent = new Agent({
  name: "分析エージェント",
  instructions: `
あなたはファイル内容を分析・要約・インサイト抽出する専門家です。

以下のMarkdown形式で出力してください:

## サマリ（3行以内）

## 重要ポイント（箇条書き 3〜7個）

## 不明点/追加で欲しい情報（あれば）

## 次アクション（最大3つ）
`.trim(),
  model: `google/${MODEL}`,
});
