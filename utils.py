"""
utils.py
不依赖 Streamlit / OpenAI / Google Sheets 的纯函数，方便单元测试。
"""

import re
from functools import lru_cache
from urllib.parse import urlparse

from config import ALLOWED_PATH_KEYWORDS

# =========================
# 简繁转换 + 文本归一化
# =========================
@lru_cache(maxsize=None)
def _opencc(config: str):
    try:
        from opencc import OpenCC
        return OpenCC(config)
    except Exception:
        return None


def convert(text: str, config: str) -> str:
    cc = _opencc(config)
    return cc.convert(text) if cc else text


def normalize(text: str) -> str:
    text = convert(text, 't2s')  # 繁体转简体，统一字体再比对
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# =========================
# 模糊匹配（rapidfuzz）
# =========================
FUZZY_THRESHOLD = 75
FUZZY_MIN_LEN_RATIO = 0.6   # 长度差太多的不算匹配（防止"宿舍"命中"宿舍可以养宠物吗"）


def best_fuzzy_match(query: str, pairs: list[dict]) -> str | None:
    """pairs: [{"question", "answer", "normalized"}]，返回最佳匹配的答案或 None"""
    from rapidfuzz import process, fuzz
    if not pairs:
        return None
    normalized_query = normalize(query)
    if not normalized_query:
        return None
    choices = {p["normalized"]: p["answer"] for p in pairs}
    results = process.extract(
        normalized_query, list(choices.keys()), scorer=fuzz.WRatio,
        score_cutoff=FUZZY_THRESHOLD, limit=5,
    )
    for cand, _, _ in results:
        short, long_ = sorted((len(normalized_query), len(cand)))
        if long_ and short / long_ >= FUZZY_MIN_LEN_RATIO:
            return choices[cand]
    return None


# =========================
# UNANSWERABLE 标记识别
# =========================
UNANSWERABLE_MARKER = "UNANSWERABLE"


def is_unanswerable(answer: str) -> bool:
    """容忍模型输出 "UNANSWERABLE." / "**UNANSWERABLE**" 之类的变体"""
    return re.sub(r'[^A-Za-z]', '', answer).upper() == UNANSWERABLE_MARKER


def may_be_unanswerable(partial: str) -> bool:
    """流式输出时：当前内容是否仍可能是 UNANSWERABLE 标记的开头"""
    if not re.fullmatch(r'[\s*`"\'.]*[A-Za-z]*[\s*`"\'.]*', partial):
        return False
    return UNANSWERABLE_MARKER.startswith(re.sub(r'[^A-Za-z]', '', partial).upper())


# =========================
# 爬虫 / secrets 辅助
# =========================
def fix_creds_json(raw: str) -> str:
    """修复 private_key 里的真实换行符，使 json.loads 能正常解析（用于本地 toml 路径）"""
    return re.sub(
        r'("private_key"\s*:\s*")(.*?)(")',
        lambda m: m.group(1) + m.group(2).replace('\n', '\\n') + m.group(3),
        raw, flags=re.DOTALL
    )


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if "cwts.edu" not in parsed.netloc:
        return False
    path = parsed.path.lower()
    return any(kw in path for kw in ALLOWED_PATH_KEYWORDS)
