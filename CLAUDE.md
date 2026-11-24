# CWTS AdBot — 开发须知

CWTS 招生 FAQ 聊天机器人（Streamlit + OpenAI + FAISS + Google Sheets）。

## 分支流程（必须遵守）
- `main` = 生产环境（线上部署），**不要直接在 main 上改代码或提交**。
- `dev` = 日常开发分支，所有改动都在 dev 上完成。
- `origin/v2` 是过时分支（只多一个 .devcontainer 提交），不要合并。

```bash
# 1. 开发前：切到 dev 并同步
git checkout dev && git pull

# 2. 改代码、提交到 dev
git add <files> && git commit -m "feat: ..." && git push

# 3. 上生产前：先跑测试，全部通过才能合并
.venv/bin/python -m pytest -q tests
#    建议再本地跑一遍页面确认：.venv/bin/streamlit run app.py

# 4. 合并到生产
git checkout main && git pull && git merge dev && git push

# 5. 回到 dev 继续开发
git checkout dev
```

合并到 main 和 push 到 main 属于上线操作，执行前先和用户确认。

## 部署（Streamlit Community Cloud）
- 生产 App 直接部署 `main` 分支：**push 到 main 会自动重新部署线上**。
- 不单独部署测试 App：改动在本地 `streamlit run app.py` 验证即可。
- 密钥只在 Streamlit Cloud 后台（App → Settings → Secrets）；本地运行需把后台 Secrets 复制到 `.streamlit/secrets.toml`。
- **本地务必连测试 Sheet**：本地 `secrets.toml` 里的 `GOOGLE_SHEET_ID` / `GOOGLE_SHEET_B_ID` 应指向生产 Sheet 的测试副本（副本需共享给同一个 service account），否则本地提问、报名会写进生产的 signups / pending / unanswered。线上用的是后台 Secrets，不受影响。
  - 测试 Sheet：`Testing_Admission_Chatbot_User_Signup`（tab `signups`）、`Testing_CWTS_AdBot_Knowledge`（tab `qa_bank` / `pending` / `unanswered`）。
  - 生产 Sheet：`Admission_Chatbot_User_Signup`、`CWTS_AdBot_Knowledge`。**不要在本地连生产 Sheet 做测试。**
- service account：`admission-bot@admissionchatbot-471103.iam.gserviceaccount.com`，新 Sheet 需共享给它（编辑者）。
- Sheet 写入按第一行表头名字对齐（`utils.build_row`），staff 可以加列、调列顺序；但不要改已有列的表头名。
- **OpenAI 余额**：余额用完时所有需要 GPT 的问题都会答不了（线上表现为"暂时无法回答"）。在 platform.openai.com → Settings → Billing 充值，建议开自动充值 / 余额提醒。

## 回答质量基准测试
- `.venv/bin/python scripts/bench_answers.py`：用真实 OpenAI + 测试 Sheet 的 qa_bank 跑一组典型问题（近义改写、只差一个词的易混问题、只有官网有答案、确实答不了），打印回答/耗时/写入哪个 tab。不写任何 Sheet，会花少量 OpenAI 额度。
- 改了检索、prompt、模糊匹配后都应跑一遍；对比 main：`git worktree add /tmp/main_wt main` 后 `.venv/bin/python scripts/bench_answers.py /tmp/main_wt main`。

## 索引重建
- `scraper.py` 爬 cwts.edu 并重建 `faiss_index/`（官网）和 `faiss_index_qa/`（qa_bank 后备），两者都提交在 git 里；线上每 24 小时后台重建一次。
- 抓到的页面少于 `MIN_PAGES` 时视为异常，保留旧索引。手动重建：`.venv/bin/python scraper.py`（需真实 OpenAI key）。

## CI（GitHub Actions）
- `.github/workflows/tests.yml`：push 到 dev / main、或向 main 发 PR 时自动跑 pytest。
- 合并 main 前先确认 dev 最新一次 CI 是绿的（GitHub 仓库 → Actions，或 `gh run list --branch dev`）。

## 环境
- 系统自带 Python 3.9 太旧（代码用了 `str | None`、`tomllib`），用项目里的 `.venv`（Python 3.12）。
- 重建环境：`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt`
- 密钥在 `.streamlit/secrets.toml`（已 gitignore，不要提交）。

## 代码结构
- `app.py` — Streamlit 页面与问答主流程（导入即执行 UI，不适合直接单测）
- `utils.py` — 不依赖外部服务的纯函数（归一化、模糊匹配、UNANSWERABLE 识别、URL 过滤），**新的纯逻辑优先放这里并补测试**
- `scraper.py` — 爬取 cwts.edu 并重建 FAISS 索引
- `sheets.py` — Google Sheets 读写
- `config.py` — 可配置项（种子 URL、阈值、模型等）
- `tests/` — pytest 测试
