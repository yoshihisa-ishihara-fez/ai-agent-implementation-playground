import { Agent } from "@mastra/core/agent";
import { listFilesTool, readFileTool } from "../tools/file-tools.js";

const MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-pro";

export const fileAgent = new Agent({
  name: "ファイル操作エージェント",
  instructions: `
あなたはinputs/ ディレクトリのファイルを一覧表示・読み取りする専門家です。

まずlist-filesでファイル一覧を取得し、重要そうなファイルを最大3つ選んでください。
選んだファイルをread-fileで1つずつ読み取り、内容をそのまま報告してください。
`.trim(),
  model: `google/${MODEL}`,
  tools: { listFilesTool, readFileTool },
});
