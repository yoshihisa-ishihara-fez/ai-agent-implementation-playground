/**
 * Mastraインスタンス
 *
 * すべてのエージェント・ワークフローをここに登録する。
 * `mastra dev` コマンドでローカルPlayground UI（localhost:4111）が起動する。
 */

import { Mastra } from "@mastra/core";
import { ragAgent } from "./agents/rag-agent.js";
import { fileAgent } from "./agents/file-agent.js";
import { analystAgent } from "./agents/analyst-agent.js";
import { supervisorAgent } from "./agents/supervisor.js";
import { ingestWorkflow } from "./workflows/ingest.js";

export const mastra = new Mastra({
  agents: {
    ragAgent,
    fileAgent,
    analystAgent,
    supervisorAgent,
  },
  workflows: {
    ingestWorkflow,
  },
});
