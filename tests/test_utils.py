import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (  # noqa: E402
    best_fuzzy_match,
    convert,
    fix_creds_json,
    is_allowed_url,
    is_unanswerable,
    may_be_unanswerable,
    normalize,
)


def _pairs(*qa):
    return [{"question": q, "answer": a, "normalized": normalize(q)} for q, a in qa]


# ---------- 简繁转换 / 归一化 ----------
def test_convert_t2s_and_s2t():
    assert convert("申請獎學金", "t2s") == "申请奖学金"
    assert convert("申请奖学金", "s2t") == "申請獎學金"


def test_normalize_unifies_traditional_and_simplified():
    assert normalize("如何申請獎學金？") == normalize("如何申请奖学金")


def test_normalize_strips_punctuation_case_and_whitespace():
    assert normalize("  How   to APPLY?! ") == "how to apply"


# ---------- 模糊匹配 ----------
QA = _pairs(
    ("如何申请奖学金", "ANS_SCHOLARSHIP"),
    ("宿舍可以养宠物吗", "ANS_PETS"),
    ("How do I apply for a scholarship", "ANS_EN_SCHOLARSHIP"),
)


def test_fuzzy_exact_match():
    assert best_fuzzy_match("如何申请奖学金", QA) == "ANS_SCHOLARSHIP"


def test_fuzzy_traditional_query_matches_simplified_question():
    assert best_fuzzy_match("如何申請獎學金？", QA) == "ANS_SCHOLARSHIP"


def test_fuzzy_english_match():
    assert best_fuzzy_match("how do i apply for scholarship", QA) == "ANS_EN_SCHOLARSHIP"


def test_fuzzy_short_query_does_not_hit_long_question():
    # 防止"宿舍"命中"宿舍可以养宠物吗"
    assert best_fuzzy_match("宿舍", QA) is None


def test_fuzzy_unrelated_query_returns_none():
    assert best_fuzzy_match("图书馆几点开门", QA) is None


@pytest.mark.parametrize("query", ["", "   ", "？！"])
def test_fuzzy_empty_query_returns_none(query):
    assert best_fuzzy_match(query, QA) is None


def test_fuzzy_empty_bank_returns_none():
    assert best_fuzzy_match("如何申请奖学金", []) is None


# ---------- UNANSWERABLE 识别 ----------
@pytest.mark.parametrize("text", ["UNANSWERABLE", "UNANSWERABLE.", "**UNANSWERABLE**", " unanswerable\n", "UNANSWERABLE!"])
def test_is_unanswerable_variants(text):
    assert is_unanswerable(text)


@pytest.mark.parametrize("text", ["", "The deadline is UNANSWERABLE soon", "申请截止日期是3月1日",
                                  "UNANSWERABLE! Sorry, try the office"])
def test_is_unanswerable_rejects_real_answers(text):
    assert not is_unanswerable(text)


@pytest.mark.parametrize("partial", ["", "U", "UNANS", "**UNANSWER", "UNANSWERABLE",
                                     "UNANSWERABLE!", "**UNANSWERABLE!**", "UNANSWERABLE?", "UNANSWERABLE！"])
def test_may_be_unanswerable_prefixes(partial):
    assert may_be_unanswerable(partial)


@pytest.mark.parametrize("partial", ["The", "申请", "UNANSWERABLE because", "UNX"])
def test_may_be_unanswerable_rejects_real_text(partial):
    assert not may_be_unanswerable(partial)


# ---------- 爬虫 URL 过滤 ----------
@pytest.mark.parametrize("url", [
    "https://www.cwts.edu/zh/admissions/application-procedure/",
    "https://www.cwts.edu/zh/academic/faculty/",
    "https://cwts.edu/ZH/About/history/",
    "https://WWW.CWTS.EDU:443/zh/admissions/",  # 大写 + 端口
])
def test_allowed_urls(url):
    assert is_allowed_url(url)


@pytest.mark.parametrize("url", [
    "https://www.cwts.edu/en/admissions/",      # 非中文路径
    "https://www.cwts.edu/zh/news/",            # 不在白名单
    "https://www.google.com/zh/admissions/",    # 外站
    "https://cwts.edu.evil.com/zh/admissions/", # 假冒子串
    "https://evilcwts.edu/zh/admissions/",      # 后缀相似
    "https://cwts.edu@evil.com/zh/admissions/", # user@ 伪装，实际访问 evil.com
    "mailto:admissions@cwts.edu",               # 非 http(s)
    "ftp://www.cwts.edu/zh/admissions/",
])
def test_disallowed_urls(url):
    assert not is_allowed_url(url)


# ---------- secrets 换行修复 ----------
def test_fix_creds_json_escapes_newlines_in_private_key():
    raw = '{"type": "service_account", "private_key": "-----BEGIN KEY-----\nabc\n-----END KEY-----\n"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    fixed = json.loads(fix_creds_json(raw))
    assert fixed["private_key"] == "-----BEGIN KEY-----\nabc\n-----END KEY-----\n"
    assert fixed["type"] == "service_account"
