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


def fuzzy_candidates(query: str, pairs: list[dict], limit: int = 3) -> list[dict]:
    """
    pairs: [{"question", "answer", "normalized"}]
    返回字面上相近的候选问答（按相似度从高到低，最多 limit 个）。
    注意：字面相近不代表是同一个问题（"养狗" vs "养猫"），候选需交给 GPT 判断后再作答。
    """
    from rapidfuzz import process, fuzz
    if not pairs:
        return []
    normalized_query = normalize(query)
    if not normalized_query:
        return []
    choices = {p["normalized"]: p for p in pairs}
    results = process.extract(
        normalized_query, list(choices.keys()), scorer=fuzz.WRatio,
        score_cutoff=FUZZY_THRESHOLD, limit=5,
    )
    out = []
    for cand, _, _ in results:
        short, long_ = sorted((len(normalized_query), len(cand)))
        if long_ and short / long_ >= FUZZY_MIN_LEN_RATIO:
            out.append(choices[cand])
    return out[:limit]


def best_fuzzy_match(query: str, pairs: list[dict]) -> str | None:
    """返回最相近候选的答案或 None"""
    cands = fuzzy_candidates(query, pairs, limit=1)
    return cands[0]["answer"] if cands else None


# =========================
# UNANSWERABLE 标记识别
# =========================
UNANSWERABLE_MARKER = "UNANSWERABLE"


_LEADING_WORD = re.compile(r'[\s*`"\'.!?！？。#_]*([A-Za-z]+)')


def is_unanswerable(answer: str) -> bool:
    """
    回复以 UNANSWERABLE 标记开头就算无法回答。
    容忍 "UNANSWERABLE." / "**UNANSWERABLE**" / "UNANSWERABLE. Please contact…" 等变体。
    """
    m = _LEADING_WORD.match(answer)
    return bool(m) and m.group(1).upper() == UNANSWERABLE_MARKER


def may_be_unanswerable(partial: str) -> bool:
    """流式输出时：当前内容是否仍可能是（或已经是）UNANSWERABLE 标记，是则先不显示"""
    if is_unanswerable(partial):
        return True  # 已确定以标记开头，后面跟什么都不显示
    # 标记前后允许的符号：空白、markdown 符号、引号、半角/全角的 . ! ?
    if not re.fullmatch(r'[\s*`"\'.!?！？。]*[A-Za-z]*[\s*`"\'.!?！？。]*', partial):
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


ALLOWED_DOMAIN = "cwts.edu"


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    # hostname 已去掉端口和 user@，且为小写；必须是 cwts.edu 本身或其子域名
    host = parsed.hostname or ""
    if host != ALLOWED_DOMAIN and not host.endswith("." + ALLOWED_DOMAIN):
        return False
    path = parsed.path.lower()
    return any(kw in path for kw in ALLOWED_PATH_KEYWORDS)
