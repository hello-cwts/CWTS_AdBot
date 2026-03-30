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


def ensure_signups_headers(ws):
    """确保 signups tab 有正确的列名（兼容旧数据）"""
    headers = [
        "conversation_id", "timestamp", "lang",
        "first_name", "last_name", "program",
        "email", "phone_or_weixin", "consent"
    ]
    existing = ws.row_values(1)
    if existing != headers:
        ws.insert_row(headers, index=1)


def append_signup_row(conversation_id: str, lang: str, first_name: str,
                      last_name: str, program: str, email: str,
                      phone_or_weixin: str, consent: bool) -> tuple[bool, str]:
    """写入用户注册记录到 Sheet A"""
    try:
        sh = get_sheet_a()
        try:
            ws = sh.worksheet(SIGNUP_SHEET_TAB)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=SIGNUP_SHEET_TAB, rows=2000, cols=20)
            ensure_signups_headers(ws)

        ws.append_row([
            conversation_id,
            datetime.now(timezone.utc).isoformat(),
            lang,
            first_name,
            last_name,
            program,
            email,
            phone_or_weixin or "",
            "yes" if consent else "no",
        ])
        return True, ""
    except Exception as e:
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
    每小时缓存一次，staff更新后最多1小时生效。
    """
    try:
        sh = get_sheet_b()
        ws = sh.worksheet(QA_SHEET_TAB)
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=["question", "answer"])
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame(columns=["question", "answer"])


def append_pending_row(conversation_id: str, lang: str, first_name: str,
                       last_name: str, email: str, program: str,
                       question: str, gpt_answer: str) -> None:
    """
    GPT成功回答后，写入 pending tab 等待staff审核。
    失败静默处理，不影响用户体验。
    """
    try:
        sh = get_sheet_b()
        try:
            ws = sh.worksheet(PENDING_SHEET_TAB)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=PENDING_SHEET_TAB, rows=2000, cols=20)
            ws.append_row([
                "conversation_id", "timestamp", "lang",
                "first_name", "last_name", "email", "program",
                "question", "gpt_answer", "status"
            ])

        ws.append_row([
            conversation_id,
            datetime.now(timezone.utc).isoformat(),
            lang,
            first_name,
            last_name,
            email,
            program,
            question,
            gpt_answer,
            "待审",
        ])
    except Exception:
        pass


def append_unanswered_row(conversation_id: str, lang: str, first_name: str,
                          last_name: str, email: str, phone_or_weixin: str,
                          program: str, question: str, gpt_answer: str) -> None:
    """
    GPT无法回答时，写入 unanswered tab，staff需跟进并联系用户。
    """
    try:
        sh = get_sheet_b()
        try:
            ws = sh.worksheet(UNANSWERED_SHEET_TAB)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=UNANSWERED_SHEET_TAB, rows=2000, cols=20)
            ws.append_row([
                "conversation_id", "timestamp", "lang",
                "first_name", "last_name", "email", "phone_or_weixin",
                "program", "question", "gpt_answer", "status", "staff_notes"
            ])

        ws.append_row([
            conversation_id,
            datetime.now(timezone.utc).isoformat(),
            lang,
            first_name,
            last_name,
            email,
            phone_or_weixin or "",
            program,
            question,
            gpt_answer or "",
            "待处理",
            "",  # staff_notes 留空
        ])
    except Exception:
        pass