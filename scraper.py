"""
scraper.py
每天自动抓取 CWTS 招生相关页面，重建 FAISS 向量库。
运行方式：
  - 手动：python scraper.py
  - 自动：用 Windows Task Scheduler 每天定时运行
"""

import re
import time
import json
import tomllib
import requests
import pandas as pd
import gspread
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from google.oauth2.service_account import Credentials
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    SEED_URLS, ALLOWED_PATH_KEYWORDS, MAX_PAGES, REQUEST_DELAY,
    FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
)


# =========================
# 工具函数
# =========================
def fix_creds_json(raw: str) -> str:
    """修复 private_key 里的真实换行符，使 json.loads 能正常解析"""
    return re.sub(
        r'("private_key"\s*:\s*")(.*?)(")',
        lambda m: m.group(1) + m.group(2).replace('\n', '\\n') + m.group(3),
        raw, flags=re.DOTALL
    )


# =========================
# secrets 读取（兼容命令行和 Streamlit）
# =========================
def load_secrets() -> dict:
    """读取 secrets，兼容 Streamlit、命令行、GitHub Actions 三种环境"""
    import os

    # 优先从环境变量读（GitHub Actions）
    if os.environ.get("OPENAI_API_KEY"):
        creds = os.environ["GOOGLE_SHEET_CREDS"].replace("\\n", "\n")
        creds = fix_creds_json(creds)  # 保险处理
        return {
            "OPENAI_API_KEY":     os.environ["OPENAI_API_KEY"],
            "GOOGLE_SHEET_B_ID":  os.environ["GOOGLE_SHEET_B_ID"],
            "GOOGLE_SHEET_CREDS": creds,
        }

    # 其次从 Streamlit secrets 读（Streamlit 会把 JSON 自动解析成 dict，保持原样）
    try:
        import streamlit as st
        return {
            "OPENAI_API_KEY":     st.secrets["OPENAI_API_KEY"],
            "GOOGLE_SHEET_B_ID":  st.secrets["GOOGLE_SHEET_B_ID"],
            "GOOGLE_SHEET_CREDS": st.secrets["GOOGLE_SHEET_CREDS"],
        }
    except Exception:
        pass

    # 最后从本地 secrets.toml 读（本地命令行）
    with open(".streamlit/secrets.toml", "rb") as f:
        s = tomllib.load(f)
    return {
        "OPENAI_API_KEY":     s["OPENAI_API_KEY"],
        "GOOGLE_SHEET_B_ID":  s["GOOGLE_SHEET_B_ID"],
        "GOOGLE_SHEET_CREDS": fix_creds_json(s["GOOGLE_SHEET_CREDS"]),
    }


# =========================
# 网站爬取
# =========================
def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if "cwts.edu" not in parsed.netloc:
        return False
    path = parsed.path.lower()
    return any(kw in path for kw in ALLOWED_PATH_KEYWORDS)


def fetch_page(url: str) -> tuple[str | None, list[str]]:
    """返回 (正文文本, 页内链接列表)，失败时返回 (None, [])"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CWTSBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = [
            urljoin(url, a["href"]).split("#")[0].split("?")[0]
            for a in soup.find_all("a", href=True)
        ]

        for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return None, links
        text = main.get_text(separator="\n", strip=True)
        return (text if len(text) >= 200 else None), links
    except Exception as e:
        print(f"  [跳过] {url} — {e}")
        return None, []


def crawl(seed_urls: list[str], max_pages: int = MAX_PAGES) -> list[Document]:
    visited = set()
    queue = list(seed_urls)
    documents = []

    print(f"开始抓取，种子页面数：{len(seed_urls)}")

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"  抓取 ({len(visited)}/{max_pages}): {url}")
        text, links = fetch_page(url)

        if text:
            documents.append(Document(page_content=text, metadata={"source": url}))
            for full_url in links:
                if full_url not in visited and is_allowed_url(full_url):
                    queue.append(full_url)

        time.sleep(REQUEST_DELAY)

    print(f"抓取完成，共 {len(documents)} 个有效页面")
    return documents


# =========================
# 向量库构建
# =========================
def build_faiss_index(documents: list[Document], openai_api_key: str) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"切段完成，共 {len(chunks)} 个段落")

    print("正在计算向量（网站内容）...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=EMBEDDING_MODEL)
    FAISS.from_documents(chunks, embeddings).save_local(FAISS_INDEX_PATH)
    print(f"网站向量库已保存到 {FAISS_INDEX_PATH}/")
    return len(chunks)


def build_qa_faiss_index(openai_api_key: str, secrets: dict) -> int:
    print("正在读取 qa_bank...")
    try:
        SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds_raw = secrets["GOOGLE_SHEET_CREDS"]
        if isinstance(creds_raw, dict):
            creds_dict = creds_raw  # Streamlit 已自动解析成 dict
        elif isinstance(creds_raw, str):
            fixed = re.sub(
                r'("private_key"\s*:\s*")(.*?)(")',
                lambda m: m.group(1) + m.group(2).replace('\n', '\\n') + m.group(3),
                creds_raw, flags=re.DOTALL
            )
            creds_dict = json.loads(fixed)
        credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        client = gspread.authorize(credentials)
        ws = client.open_by_url(secrets["GOOGLE_SHEET_B_ID"]).worksheet("qa_bank")
        df = pd.DataFrame(ws.get_all_records())
    except Exception as e:
        print(f"  读取 qa_bank 失败：{e}")
        return 0

    if df.empty:
        print("  qa_bank 为空，跳过建库。")
        return 0

    docs = [
        Document(
            page_content=f"Q: {r['question']}\nA: {r['answer']}",
            metadata={"source": "qa_bank"}
        )
        for _, r in df.iterrows()
        if r.get("question") and r.get("answer")
    ]

    if not docs:
        print("  qa_bank 没有有效条目，跳过建库。")
        return 0

    print(f"  正在为 qa_bank {len(docs)} 条建向量库...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=EMBEDDING_MODEL)
    FAISS.from_documents(docs, embeddings).save_local("faiss_index_qa")
    print(f"  qa_bank 向量库已保存到 faiss_index_qa/")
    return len(docs)


# =========================
# 主入口
# =========================
def rebuild_index(openai_api_key: str) -> bool:
    print("=== 开始重建向量库 ===")
    secrets = load_secrets()

    # 1) 网站向量库
    docs = crawl(SEED_URLS)
    if not docs:
        print("未抓到任何内容，终止。")
        return False
    n = build_faiss_index(docs, openai_api_key)
    print(f"网站向量库完成，共 {n} 个段落")

    # 2) qa_bank 向量库
    n_qa = build_qa_faiss_index(openai_api_key, secrets)
    print(f"qa_bank 向量库完成，共 {n_qa} 条")

    print("=== 重建完成 ===")
    return True


if __name__ == "__main__":
    secrets = load_secrets()
    rebuild_index(secrets["OPENAI_API_KEY"])