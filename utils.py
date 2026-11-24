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
NON_HTML_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                       ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".mp3", ".mp4")


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    # hostname 已去掉端口和 user@，且为小写；必须是 cwts.edu 本身或其子域名
    host = parsed.hostname or ""
    if host != ALLOWED_DOMAIN and not host.endswith("." + ALLOWED_DOMAIN):
        return False
    path = parsed.path.lower().rstrip("/")
    if path.endswith(NON_HTML_EXTENSIONS):
        return False
    # 按路径段前缀匹配："/zh/about" 匹配 "/zh/about" 和 "/zh/about/..."，
    # 不匹配 "/en/zh/about" 或 "/zh/aboutus-old"
    return any(path == kw or path.startswith(kw + "/") for kw in ALLOWED_PATH_KEYWORDS)


def normalize_url(url: str) -> str:
    """
    爬虫去重用：同一页面的不同写法归一成一个网址。
    小写 scheme/host、去掉默认端口、# 锚点、? 参数和末尾斜杠
    （"…/tuition-scholarship" 与 "…/tuition-scholarship/" 视为同一页）。
    """
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{host}{path}"


# =========================
# 显示用户输入
# =========================
_MD_SPECIAL = re.compile(r'([\\`*_{}\[\]()#+\-.!|<>~])')


def escape_markdown(text: str) -> str:
    """把用户输入原样显示在 st.markdown 里：转义 markdown/HTML 特殊符号，换行合并成空格"""
    return _MD_SPECIAL.sub(r'\\\1', " ".join(text.split()))


# =========================
# Google Sheet：按表头名字排列一行
# =========================
def _header_key(h: str) -> str:
    return str(h).strip().lower()


def build_row(headers: list[str], values: dict) -> tuple[list, list[str]]:
    """
    按 Sheet 第一行的表头名字排列要写入的值，避免列顺序变动/staff 加列导致错位。
    表头匹配忽略大小写和首尾空格；values 里没有的列留空。
    返回 (row, missing)：missing 是表头里找不到的字段名（这些值没写进去，调用方应打日志）。
    """
    index = {}
    for i, h in enumerate(headers):
        index.setdefault(_header_key(h), i)
    # 去掉末尾的空表头，避免整行写得过长
    width = len(headers)
    while width and not str(headers[width - 1]).strip():
        width -= 1
    row = [""] * width
    missing = []
    for k, v in values.items():
        i = index.get(_header_key(k))
        if i is None or i >= width:
            missing.append(k)
        else:
            row[i] = "" if v is None else v
    return row, missing
