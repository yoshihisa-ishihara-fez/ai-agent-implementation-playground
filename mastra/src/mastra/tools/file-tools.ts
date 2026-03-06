import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";

const INPUTS_DIR = "inputs";

export const listFilesTool = createTool({
  id: "list-files",
  description:
    "inputs/ ディレクトリ内のファイル一覧を返す。ファイルが存在しない場合はその旨を返す。",
  inputSchema: z.object({}),
  outputSchema: z.object({ files: z.array(z.string()) }),
  execute: async () => {
    if (!fs.existsSync(INPUTS_DIR)) {
      return { files: [] as string[] };
    }
    const files = fs
      .readdirSync(INPUTS_DIR)
      .filter((f) => {
        const stat = fs.statSync(path.join(INPUTS_DIR, f));
        return stat.isFile() && !f.startsWith(".");
      })
      .sort();
    return { files };
  },
});

export const readFileTool = createTool({
  id: "read-file",
  description:
    "inputs/ ディレクトリ内の指定ファイルの内容を返す。パス区切り文字なしのファイル名で指定すること（例: sample1.txt）。",
  inputSchema: z.object({
    filename: z
      .string()
      .describe("ファイル名（パス区切り文字なし、例: sample1.txt）"),
  }),
  outputSchema: z.object({ content: z.string() }),
  execute: async ({ context: { filename: rawFilename } }) => {
    const filename = rawFilename.trim();

    if (!filename || filename.includes("/") || filename.includes("\\") || filename.startsWith(".")) {
      return { content: `エラー: 不正なファイル名です: ${filename}` };
    }

    const filePath = path.join(INPUTS_DIR, filename);

    try {
      const resolved = path.resolve(filePath);
      const inputsResolved = path.resolve(INPUTS_DIR);
      if (!resolved.startsWith(inputsResolved)) {
        return { content: `エラー: inputs/ 外のファイルは読めません: ${filename}` };
      }
    } catch {
      return { content: `エラー: パス解決に失敗しました` };
    }

    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      return { content: `エラー: ファイルが見つかりません: ${filename}` };
    }

    try {
      const content = fs.readFileSync(filePath, "utf-8").slice(0, 8000);
      return { content: content.trim() || "(空ファイル)" };
    } catch (e) {
      return { content: `エラー: 読み取りに失敗しました: ${e}` };
    }
  },
});
