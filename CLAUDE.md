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
