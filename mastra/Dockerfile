FROM node:22-slim

WORKDIR /app

ENV NODE_NO_WARNINGS=1
# モデルキャッシュの保存先（ビルド時・実行時で共通）
ENV HF_CACHE_DIR=/app/.cache/transformers

COPY package*.json ./
RUN npm ci

COPY . .

# モデルを事前ダウンロード（起動時間の短縮）
RUN node scripts/download-model.mjs

CMD ["node", "--import", "tsx/esm", "src/index.ts"]
