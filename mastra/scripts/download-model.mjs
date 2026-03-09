import { pipeline, env } from "@huggingface/transformers";

env.cacheDir = process.env.HF_CACHE_DIR ?? ".cache/transformers";

console.log(`Downloading paraphrase-multilingual-MiniLM-L12-v2 to ${env.cacheDir} ...`);
await pipeline("feature-extraction", "Xenova/paraphrase-multilingual-MiniLM-L12-v2");
console.log("Download complete.");
