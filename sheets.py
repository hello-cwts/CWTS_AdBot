"""
sheets.py
所有 Google Sheet 读写操作集中在这里。

Sheet A (GOOGLE_SHEET_ID)   → signups
Sheet B (GOOGLE_SHEET_B_ID) → qa_bank / pending / unanswered
"""

import json
import random
import string
from datetime import datetime, timezone

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

from utils import build_row
from config import (
    SIGNUP_SHEET_TAB,
    QA_SHEET_TAB,
    PENDING_SHEET_TAB,
    UNANSWERED_SHEET_TAB,
)

# =========================
# Auth
# =========================
@st.cache_resource(ttl=3600)
def get_gs_client():
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDS"])
    credentials = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(credentials)


# =========================
# Conversation ID
# =========================
def generate_conversation_id() -> str:
    """生成唯一对话ID，格式：conv_20260328_a3f7"""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"conv_{date_str}_{suffix}"


def get_or_create_conversation_id() -> str:
    """从 session_state 获取或新建 conversation_id"""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = generate_conversation_id()
    return st.session_state.conversation_id


# =========================
# Sheet A：用户注册
# =========================
def get_sheet_a():
    client = get_gs_client()
    return client.open_by_url(st.secrets["GOOGLE_SHEET_ID"])


SIGNUP_HEADERS = [
    "conversation_id", "timestamp", "lang",
    "first_name", "last_name", "program",
    "email", "phone_or_weixin", "consent",
]
PENDING_HEADERS = [
    "conversation_id", "timestamp", "lang",
    "first_name", "last_name", "email", "phone_or_weixin", "program",
    "question", "gpt_answer", "status",
]
UNANSWERED_HEADERS = [
    "conversation_id", "timestamp", "lang",
    "first_name", "last_name", "email", "phone_or_weixin",
    "program", "question", "gpt_answer", "status", "staff_notes",
]


def _get_or_create_tab(sh, title: str, headers: list[str]):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=2000, cols=max(20, len(headers)))
        ws.append_row(headers)
        return ws


def _append_by_header(ws, default_headers: list[str], values: dict) -> None:
    """
    按第一行表头的名字写入，staff 调整列顺序或加列都不会错位。
    第一行是空的就先写入默认表头。
    """
    headers = ws.row_values(1)
    if not any(str(h).strip() for h in headers):
        ws.update([default_headers], "A1")
        headers = default_headers
    row, missing = build_row(headers, values)
    if missing:
        print(f"[{ws.title}] 表头里没有这些列，对应的值未写入：{missing}")
    ws.append_row(row, table_range="A1")


def append_signup_row(conversation_id: str, lang: str, first_name: str,
                      last_name: str, program: str, email: str,
                      phone_or_weixin: str, consent: bool) -> tuple[bool, str]:
    """写入用户注册记录到 Sheet A"""
    try:
        ws = _get_or_create_tab(get_sheet_a(), SIGNUP_SHEET_TAB, SIGNUP_HEADERS)
        _append_by_header(ws, SIGNUP_HEADERS, {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lang": lang,
            "first_name": first_name,
            "last_name": last_name,
            "program": program,
            "email": email,
            "phone_or_weixin": phone_or_weixin or "",
            "consent": "yes" if consent else "no",
        })
        return True, ""
    except Exception as e:
        print(f"[append_signup_row 写入失败] conv={conversation_id} {type(e).__name__}: {e}")
        return False, str(e)


# =========================
# Sheet B：知识库
# =========================
def get_sheet_b():
    client = get_gs_client()
    return client.open_by_url(st.secrets["GOOGLE_SHEET_B_ID"])


@st.cache_data(ttl=300)
def load_qa_bank() -> pd.DataFrame:
    """
    读取 qa_bank tab，返回 DataFrame。
    缓存 5 分钟，staff 更新后最多 5 分钟对精确匹配生效
    （语义检索用的 faiss_index_qa 要等下次 24 小时重建）。
    """
    try:
        sh = get_sheet_b()
        ws = sh.worksheet(QA_SHEET_TAB)
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=["question", "answer"])
        return pd.DataFrame(records)
    except Exception as e:
        # 不返回空表：st.cache_data 不缓存异常，下次请求会重试；
        # 返回空表会被缓存 5 分钟，期间 Q&A 库悄悄失效。调用方负责兜底。
        print(f"[load_qa_bank 读取失败] {type(e).__name__}: {e}")
        raise


def append_pending_row(conversation_id: str, lang: str, first_name: str,
                       last_name: str, email: str, program: str,
                       question: str, gpt_answer: str,
                       phone_or_weixin: str = "") -> None:
    """
    GPT成功回答后，写入 pending tab 等待staff审核。
    失败只打日志（Streamlit Cloud → Manage app → Logs 可见），不影响用户体验。
    """
    try:
        ws = _get_or_create_tab(get_sheet_b(), PENDING_SHEET_TAB, PENDING_HEADERS)
        _append_by_header(ws, PENDING_HEADERS, {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lang": lang,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone_or_weixin": phone_or_weixin or "",
            "program": program,
            "question": question,
            "gpt_answer": gpt_answer,
            "status": "待审",
        })
    except Exception as e:
        print(f"[append_pending_row 写入失败] conv={conversation_id} {type(e).__name__}: {e}")


def append_unanswered_row(conversation_id: str, lang: str, first_name: str,
                          last_name: str, email: str, phone_or_weixin: str,
                          program: str, question: str, gpt_answer: str) -> None:
    """
    GPT无法回答时，写入 unanswered tab，staff需跟进并联系用户。
    """
    try:
        ws = _get_or_create_tab(get_sheet_b(), UNANSWERED_SHEET_TAB, UNANSWERED_HEADERS)
        _append_by_header(ws, UNANSWERED_HEADERS, {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lang": lang,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone_or_weixin": phone_or_weixin or "",
            "program": program,
            "question": question,
            "gpt_answer": gpt_answer or "",
            "status": "待处理",
            "staff_notes": "",
        })
    except Exception as e:
        print(f"[append_unanswered_row 写入失败] conv={conversation_id} {type(e).__name__}: {e}")
