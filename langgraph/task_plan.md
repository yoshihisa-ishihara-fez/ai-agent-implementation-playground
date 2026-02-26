# LangGraph Todo-Planner タスクプラン

## 1. LangGraph とは何か

LangGraph は LangChain チームが開発した、**グラフベースのエージェントオーケストレーションフレームワーク**。
エージェントのワークフローを「ノード（処理）」と「エッジ（遷移）」のグラフとして定義し、
ループや条件分岐を含む複雑なフローを明示的に制御できる。

### コアコンセプト

- **StateGraph**: 状態を持つ有向グラフ。ノード間で共有される State を定義する
- **Node**: グラフ上の処理単位（LLM 呼び出し、ツール実行など）
- **Edge**: ノード間の遷移。条件付きエッジ（conditional edge）でフロー分岐が可能
- **State**: TypedDict や Pydantic BaseModel で定義する共有データ構造

---

## 2. LangChain 単体と比べて何が良いのか

### LangChain 単体（現在の実装）の特徴

- `create_agent()` でエージェントを生成し、内部のツール呼び出しループは**フレームワーク任せ**
- コードがシンプルで立ち上げが速い
- ループや分岐の制御はブラックボックス

### LangGraph を使うメリット

| 観点 | LangChain 単体 | LangGraph |
|---|---|---|
| **フロー制御** | 暗黙的（フレームワーク内部） | 明示的（グラフで定義） |
| **状態管理** | 限定的（messages のみ） | 組み込み State で自由に設計可能 |
| **デバッグ** | ループ内部が見えにくい | ノード単位でステップ実行・ログ取得可能 |
| **拡張性** | ツール追加程度 | ノード追加、条件分岐、並列実行、サブグラフ |
| **Human-in-the-loop** | 自前実装が必要 | interrupt / resume が組み込み |
| **永続化** | なし | チェックポインタによる状態の保存・復元 |

### 学習面でのメリット

- エージェントの「思考 → 行動 → 観察」ループが**コードとして可視化**される
- LangChain の `create_agent()` が内部で何をしているかを理解できる
- 将来的にマルチエージェント構成へ拡張する基盤になる

---

## 3. LangGraph の課題・注意点

- **学習コスト**: グラフ定義・State 設計など、概念が増える
- **ボイラープレート**: 単純なエージェントでもノード・エッジの定義が必要でコード量が増える
- **バージョン変化**: API が変わりやすい（2025年末に v1.0 到達で安定化）
- **過剰設計のリスク**: 今回のようなシンプルなエージェントには LangGraph はオーバースペックとも言える。ただし**学習目的**としては適切
- **デバッグツール依存**: LangSmith との連携が前提の部分がある（なくても動くが体験が変わる）

---

## 4. 設計方針（決定済み）

- **StateGraph フルスクラッチ**で構築する（`create_react_agent` は使わない）
- **State にカスタムフィールド**を持たせ、中間状態を明示的に管理する
- **LLM は Google Gemini** を継続利用（既存と同じ）
- LangGraph の良さを実感するため、以下の3要素を**すべて組み込む**:
  1. **Human-in-the-loop** — ファイル選択後にユーザー承認を挟む
  2. **構造化 State** — 各ステップの中間結果を State フィールドで管理
  3. **条件分岐による明示的フロー制御** — エラーハンドリング・ループをグラフで表現

### State 定義

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    file_list: list[str]           # ls_node で取得したファイル一覧
    selected_files: list[str]      # LLM が選んだファイル（最大3つ）
    approved_files: list[str]      # ユーザーが承認したファイル
    file_contents: dict[str, str]  # ファイル名 → 内容のマップ
    summary: str                   # 最終要約
```

### グラフ構造

```
START
  │
  ▼
ls_node（ファイル一覧取得）
  │
  ▼
select_node（LLM がファイル選択 → selected_files に格納）
  │
  ▼
[ファイルあり?] ── No ──→ error_node（「対象ファイルなし」）→ END
  │
  Yes
  ▼
human_review（interrupt で一時停止、ユーザーに選択内容を提示）
  │
  ▼
[承認?] ── No ──→ select_node に戻る（再選択ループ）
  │
  Yes
  ▼
read_node（approved_files を順に読み取り → file_contents に格納）
  │
  ▼
summarize_node（LLM が file_contents をもとに要約生成 → summary に格納）
  │
  ▼
END
```

### LangChain 版との対比

| 処理 | LangChain 版 | LangGraph 版 |
|---|---|---|
| ファイル一覧取得 | LLM がツール呼び出し（暗黙） | `ls_node` が実行（明示） |
| ファイル選択 | LLM が勝手に判断 | `select_node` で LLM が判断 → State に記録 |
| 選択の確認 | なし | `human_review` で interrupt → ユーザー承認 |
| ファイル読み取り | LLM がツール呼び出し（暗黙） | `read_node` が approved_files を順に読む |
| 要約生成 | LLM が最終応答として出力 | `summarize_node` が file_contents から生成 |
| エラー処理 | LLM 任せ | conditional edge で明示的に分岐 |

---

## 5. 実装タスク

### Phase 1: プロジェクトセットアップ

- [ ] `requirements.txt` 作成（langgraph, langchain-core, langchain-google-genai 等）
- [ ] `.env.example` 作成（既存を流用）
- [ ] `Dockerfile` / `docker-compose.yml` 作成
- [ ] `.gitignore` 作成
- [ ] `inputs/` ディレクトリとサンプルファイルを配置（既存からコピー or シンボリックリンク）

### Phase 2: ツール・ユーティリティ実装

- [ ] `tools.py` — ツール関数の定義
  - `ls_inputs()`: 既存ロジックを流用
  - `read_file(filename)`: 既存ロジックを流用（セキュリティチェック含む）

### Phase 3: グラフ構築（コア実装）

- [ ] `state.py` — AgentState の TypedDict 定義
- [ ] `nodes.py` — 各ノード関数の実装
  - `ls_node`: `ls_inputs` を実行し `file_list` を更新
  - `select_node`: LLM に file_list を渡してファイル選択させ `selected_files` を更新
  - `human_review_node`: `interrupt()` でユーザー承認を待つ、承認結果を `approved_files` に格納
  - `read_node`: `approved_files` のファイルを読み取り `file_contents` を更新
  - `summarize_node`: LLM に file_contents を渡して要約生成、`summary` を更新
  - `error_node`: エラーメッセージを出力
- [ ] `graph.py` — StateGraph の構築・コンパイル
  - ノード登録
  - エッジ定義（条件分岐含む）
  - `graph.compile(checkpointer=...)` でコンパイル
- [ ] `agent.py` — エントリポイント（main 関数）
  - グラフの実行
  - Human-in-the-loop のインタラクション処理（interrupt 後の resume）
  - 最終出力の表示

### Phase 4: 動作確認

- [ ] Docker で実行して正常系の動作確認
- [ ] Human-in-the-loop の承認 / 拒否フローの確認
- [ ] ファイル0件時の条件分岐（error_node への遷移）の確認
- [ ] 各ノード遷移のログ出力で、グラフの実行フローを可視化

### Phase 5: 発展（オプション）

- [ ] ストリーミング出力の実装（ノード単位で進捗を表示）
- [ ] Zenn 記事の執筆（LangChain 版との比較を中心に）

---

## 6. ファイル構成（予定）

```
langgraph/todo-planner/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── task_plan.md          ← 本ファイル
├── agent.py              ← エントリポイント
├── state.py              ← AgentState 定義
├── nodes.py              ← ノード関数群
├── tools.py              ← ツール関数群
├── graph.py              ← StateGraph 構築・コンパイル
└── inputs/
    ├── sample1.txt
    └── sample2.md
```

---

## 参考リンク

- [LangGraph 公式](https://www.langchain.com/langgraph)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [ReAct agent from scratch (LangGraph)](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [LangChain vs LangGraph 比較](https://duplocloud.com/blog/langchain-vs-langgraph/)
- [LangChain & LangGraph v1.0 リリースブログ](https://blog.langchain.com/langchain-langgraph-1dot0/)
