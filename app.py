"""
app.py
CWTS 招生FAQ Chatbot v2
"""

import threading
import streamlit as st
import re
from datetime import datetime, timezone
from pathlib import Path

from config import (
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    GPT_MODEL,
    GPT_TEMPERATURE,
)
from utils import (
    convert as _convert,
    normalize as _normalize,
    escape_markdown,
    fuzzy_candidates,
    is_near_exact,
    is_unanswerable,
    UNANSWERABLE_MARKER,
    may_be_unanswerable as _may_be_unanswerable,
)
from sheets import (
    get_or_create_conversation_id,
    load_qa_bank,
    append_signup_row,
    append_pending_row,
    append_unanswered_row,
)

# =========================
# Page config
# =========================
st.set_page_config(page_title="CWTS - 申请常见问题答疑", layout="wide")

# =========================
# Sidebar：语言选择 + 链接
# =========================
with st.sidebar:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### 🌍 語言 / Language")
    lang = st.radio("語言", ["中文(简)", "中文(繁)", "English"], index=1, label_visibility="collapsed")
    lang_code = {"中文(简)": "zh", "中文(繁)": "zh-TW", "English": "en"}[lang]

    st.write("")
    st.markdown(
        """
        <a href="https://www.cwts.edu/zh/admissions/application-procedure/"
           target="_blank"
           style="display:inline-block;font-size:14px;color:#000;text-decoration:none;
                  border:1px solid #000;padding:6px 12px;border-radius:6px;background:#fff;">
           👉 申請鏈接 / Apply Now
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <a href="mailto:admissions@cwts.edu"
           style="display:inline-block;font-size:14px;color:#000;text-decoration:none;
                  border:1px solid #000;padding:6px 12px;border-radius:6px;background:#fff;">
           💬 聯絡我們 / Contact us
        </a>
        """,
        unsafe_allow_html=True,
    )

# =========================
# 多语言文案
# =========================
T = {
    "title": {
        "zh": "CWTS - 申请常见问题答疑",
        "zh-TW": "CWTS - 申請常見問題答疑",
        "en": "CWTS - Application FAQ",
    },
    "search_prompt": {
        "zh": "🔍 请输入您的问题（例如：如何申请奖学金 / 什么是财务证明）",
        "zh-TW": "🔍 請輸入您的問題（例如：如何申請獎學金 / 什麼是財務證明）",
        "en": "🔍 Enter your question (e.g. How to apply for scholarship / What is financial proof?)",
    },
    "input_placeholder": {
        "zh": "在此输入问题，按 Enter 或点击右侧箭头发送",
        "zh-TW": "在此輸入問題，按 Enter 或點擊右側箭頭送出",
        "en": "Type your question, then press Enter or click the arrow to send",
    },
    "your_question": {
        "zh": "您的问题：",
        "zh-TW": "您的問題：",
        "en": "Your question:",
    },
    "answer_title": {
        "zh": "### 🙋 回答",
        "zh-TW": "### 🙋 回答",
        "en": "### 🙋 Answer",
    },
    "searching": {
        "zh": "🔍 正在检索相关资料…",
        "zh-TW": "🔍 正在檢索相關資料…",
        "en": "🔍 Searching for relevant information…",
    },
    "thinking": {
        "zh": "⏳ 正在思考并生成回答，请稍候…",
        "zh-TW": "⏳ 正在思考並生成回答，請稍候…",
        "en": "⏳ Thinking and generating an answer, please wait…",
    },
    "source_label": {
        "zh": "来源",
        "zh-TW": "來源",
        "en": "Source",
    },
    "no_result": {
        "zh": "暂时无法回答这个问题，我们已记录您的问题，招生团队会尽快与您联系！如有紧急需要，请发邮件至 admissions@cwts.edu",
        "zh-TW": "暫時無法回答這個問題，我們已記錄您的問題，招生團隊會盡快與您聯繫！如有緊急需要，請發郵件至 admissions@cwts.edu",
        "en": "We are unable to answer this question right now. We have recorded it and our admissions team will follow up with you shortly! For urgent enquiries, email admissions@cwts.edu",
    },
    "source_expander": {
        "zh": "🔎 参考来源（点击展开）",
        "zh-TW": "🔎 參考來源（點此展開）",
        "en": "🔎 Sources (click to expand)",
    },
}

def t(key: str) -> str:
    return T[key][lang_code]


# =========================
# OpenAI 客户端 / 向量库
# =========================
# OpenAI 出错（限流、余额不足等）时只重试 1 次，避免学生白等十几秒
OPENAI_MAX_RETRIES = 1
OPENAI_TIMEOUT = 30


@st.cache_resource
def get_openai_client():
    import openai
    return openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"],
                         max_retries=OPENAI_MAX_RETRIES, timeout=OPENAI_TIMEOUT)


@st.cache_resource
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"], model=EMBEDDING_MODEL,
                            max_retries=OPENAI_MAX_RETRIES, request_timeout=OPENAI_TIMEOUT)


@st.cache_resource(ttl=3600)
def load_faiss_store():
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local(FAISS_INDEX_PATH, get_embeddings(), allow_dangerous_deserialization=True)


@st.cache_resource(ttl=3600)
def load_qa_faiss_store():
    """磁盘上的 qa_bank 向量库，仅在读取 Sheet 失败时作为后备"""
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local("faiss_index_qa", get_embeddings(), allow_dangerous_deserialization=True)


@st.cache_resource(max_entries=1)
def _build_qa_store(pairs: tuple[tuple[str, str], ...]):
    """按 qa_bank 当前内容在内存中建向量库；内容一变（缓存 key 变）就重建，旧的自动淘汰"""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    docs = [Document(page_content=f"Q: {q}\nA: {a}", metadata={"source": "qa_bank"})
            for q, a in pairs]
    return FAISS.from_documents(docs, get_embeddings())


def load_qa_store():
    """staff 改了 qa_bank 后，语义检索也在 5 分钟内（_load_qa_pairs 的缓存周期）生效"""
    try:
        pairs = tuple((p["question"], p["answer"]) for p in _load_qa_pairs())
    except Exception:
        pairs = ()  # Sheet 读取失败（已在 load_qa_bank 打日志），用磁盘版本兜底
    if pairs:
        try:
            return _build_qa_store(pairs)
        except Exception as e:
            print(f"[qa_bank 向量库构建失败，改用磁盘版本] {e}")
    return load_qa_faiss_store()


# =========================
# 重建向量库（后台）
# =========================
REBUILD_INTERVAL_HOURS = 24
REBUILD_RETRY_HOURS = 1     # 重建失败后，至少间隔多久再重试
LAST_REBUILD_FILE = "last_rebuild.txt"


@st.cache_resource
def _rebuild_state() -> dict:
    """跨 session / rerun 共享的重建状态（脚本每次 rerun 都会重置普通全局变量）"""
    return {"lock": threading.Lock(), "last_attempt": 0.0}


def should_rebuild() -> bool:
    now = datetime.now(timezone.utc).timestamp()
    if (now - _rebuild_state()["last_attempt"]) / 3600 < REBUILD_RETRY_HOURS:
        return False
    try:
        last = float(Path(LAST_REBUILD_FILE).read_text().strip())
    except Exception:
        # Streamlit Cloud 重启后临时磁盘上的记录会丢失；改用索引文件的修改时间
        # （即部署/上次重建的时间），否则每次重启都会全站重爬、重算向量
        try:
            last = (Path(FAISS_INDEX_PATH) / "index.faiss").stat().st_mtime
        except Exception:
            return True
    return (now - last) / 3600 >= REBUILD_INTERVAL_HOURS

def mark_rebuilt():
    Path(LAST_REBUILD_FILE).write_text(str(datetime.now(timezone.utc).timestamp()))

def async_rebuild(state: dict):
    try:
        from scraper import rebuild_index
        if rebuild_index(st.secrets["OPENAI_API_KEY"]):
            mark_rebuilt()
            load_faiss_store.clear()
            load_qa_faiss_store.clear()
    except Exception as e:
        print(f"[后台重建失败] {e}")
    finally:
        state["lock"].release()

def trigger_rebuild_if_needed():
    """同一时间只允许一个重建线程"""
    if not should_rebuild():
        return
    state = _rebuild_state()
    if not state["lock"].acquire(blocking=False):
        return  # 已有线程在重建
    state["last_attempt"] = datetime.now(timezone.utc).timestamp()
    threading.Thread(target=async_rebuild, args=(state,), daemon=True).start()


# =========================
# 精确匹配缓存（rapidfuzz）
# =========================
@st.cache_data(ttl=300)
def _load_qa_pairs() -> list[dict]:
    df = load_qa_bank()
    pairs = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if q and a:
            pairs.append({"question": q, "answer": a, "normalized": _normalize(q)})
    return pairs


def fuzzy_candidates_qa(query: str) -> list[dict]:
    try:
        return fuzzy_candidates(query, _load_qa_pairs())
    except Exception:
        return []


# =========================
# 校训动画（只显示一次）
# =========================
if "verse_displayed" not in st.session_state:
    st.session_state.verse_displayed = False

if not st.session_state.verse_displayed:
    quote_area = st.empty()
    quote_area.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC&display=swap');
@keyframes fadeInOut {
  0%{opacity:0;transform:translateY(8px)}15%{opacity:1;transform:translateY(0)}85%{opacity:1}100%{opacity:0}
}
.fade-in-out{animation:fadeInOut 3.2s ease-in-out forwards}
.verse-wrap{position:fixed;inset:0;z-index:1000;pointer-events:none;display:flex;justify-content:center;align-items:center}
.verse-card{backdrop-filter:saturate(140%) blur(10px);background:rgba(255,255,255,.86);
  border:1px solid rgba(0,0,0,.06);border-radius:16px;padding:28px 36px;
  box-shadow:0 12px 32px rgba(0,0,0,.14);max-width:760px;width:min(92vw,760px);text-align:center}
.verse-text{font-family:'Noto Serif SC',serif;font-size:22px;line-height:2;color:#2f2f2f;letter-spacing:.6px}
.verse-sep{height:1px;width:70%;margin:12px auto 14px;
  background:linear-gradient(90deg,transparent,#6b4eff,transparent)}
.badge{display:inline-block;font-size:12px;letter-spacing:.18em;text-transform:uppercase;
  color:#666;border:1px solid rgba(107,78,255,.25);background:rgba(107,78,255,.08);
  padding:4px 10px;border-radius:999px;margin-bottom:10px}
</style>
<div class="verse-wrap fade-in-out">
  <div class="verse-card">
    <div class="badge">校訓</div>
    <div class="verse-text">立於聖言勤於禱<br/>終於使命敏於行</div>
    <div class="verse-sep"></div>
  </div>
</div>
""", unsafe_allow_html=True)
    # 浮层 + CSS 动画自行淡出，不再 time.sleep 阻塞页面（以前首次打开要干等 4 秒才出现表单）
    st.session_state.verse_displayed = True


# =========================
# 注册表单
# =========================
PROGRAM_OPTIONS = {
    "zh": [
        "教牧博士（DMin）", "神学硕士（ThM）", "道学硕士（MDiv）",
        "圣经研究硕士（MBS）", "基督教圣工硕士（MCM）",
        "基督教圣工硕士（网络）（MCM Online）", "跨文化领袖硕士（MCCL）",
        "神学精要硕士（MTE）", "基督教研究文凭（DCS）", "教会事工文凭（DCL）",
        "基督工人证书", "家庭事工证书", "儿童事工证书", "宣教事工证书", "不确定",
    ],
    "zh-TW": [
        "教牧博士（DMin）", "神學碩士（ThM）", "道學碩士（MDiv）",
        "聖經研究碩士（MBS）", "基督教聖工碩士（MCM）",
        "基督教聖工碩士（網路）（MCM Online）", "跨文化領袖碩士（MCCL）",
        "神學精要碩士（MTE）", "基督教研究文憑（DCS）", "教會事工文憑（DCL）",
        "基督工人證書", "家庭事工證書", "兒童事工證書", "宣教事工證書", "不確定",
    ],
    "en": [
        "Doctor of Ministry (DMin)", "Master of Theology (ThM)",
        "Master of Divinity (MDiv)", "Master of Biblical Studies (MBS)",
        "Master of Christian Ministry (MCM)", "MCM Online",
        "Master of Cross-Cultural Leadership (MCCL)",
        "Master of Theological Essentials (MTE)",
        "Diploma of Christian Studies (DCS)", "Diploma in Church Leadership (DCL)",
        "Certificate in Servant Leadership", "Certificate of Family Ministry",
        "Certificate of Children Ministry", "Certificate of Evangelism Ministry",
        "Undecided",
    ],
}

FORM_LABELS = {
    "zh":    {"title":"📝 基本信息","desc":"为了更好地协助您，请先填写以下信息。",
              "first":"名*","last":"姓*","program":"目标学位项目*",
              "email":"电子邮件*","phone":"电话或微信（选填）",
              "consent":"我同意学院以此资料联系我*","submit":"提交",
              "err_name":"请输入姓名","err_email":"请输入正确的邮箱",
              "err_consent":"请勾选同意条款","ok":"已收到，感谢！您现在可以提问啦。",
              "err_write":"提交失败，请稍后再试，或发邮件至 admissions@cwts.edu"},
    "zh-TW": {"title":"📝 基本資訊","desc":"為了更好地協助您，請先填寫以下資訊。",
              "first":"名*","last":"姓*","program":"目標學位項目*",
              "email":"電子郵件*","phone":"電話或微信（選填）",
              "consent":"我同意學院以此資料聯絡我*","submit":"提交",
              "err_name":"請輸入姓名","err_email":"請輸入正確的郵箱",
              "err_consent":"請勾選同意條款","ok":"已收到，感謝！您現在可以提問囉。",
              "err_write":"提交失敗，請稍後再試，或發郵件至 admissions@cwts.edu"},
    "en":    {"title":"📝 Basic Info","desc":"To serve you better, please fill out the form below.",
              "first":"First name*","last":"Last name*","program":"Target Degree Program*",
              "email":"Email*","phone":"Phone or Weixin (optional)",
              "consent":"I consent to be contacted by the seminary*","submit":"Submit",
              "err_name":"Please enter your name","err_email":"Please enter a valid email",
              "err_consent":"Please provide consent","ok":"Thank you! You can ask questions now.",
              "err_write":"Submission failed. Please try again later or email admissions@cwts.edu"},
}

def show_signup_form():
    lbl = FORM_LABELS[lang_code]
    if st.session_state.get("signed_up"):
        return

    st.markdown(f"### {lbl['title']}")
    st.caption(lbl["desc"])

    with st.form("signup_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        first_name = col1.text_input(lbl["first"], placeholder="e.g. 三 San", max_chars=60)
        last_name  = col2.text_input(lbl["last"],  placeholder="e.g. 张 Zhang", max_chars=60)
        email      = st.text_input(lbl["email"],   placeholder="name@example.com", max_chars=120)
        phone      = st.text_input(lbl["phone"],   placeholder="+1 650-123-4567", max_chars=60)
        program    = st.selectbox(lbl["program"],  PROGRAM_OPTIONS[lang_code])
        consent    = st.checkbox(lbl["consent"])
        submitted  = st.form_submit_button(lbl["submit"])

        if submitted:
            email_ok = re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or "")
            if not (first_name and last_name):
                st.error(lbl["err_name"])
            elif not email_ok:
                st.error(lbl["err_email"])
            elif not consent:
                st.error(lbl["err_consent"])
            else:
                conv_id = get_or_create_conversation_id()
                ok, err = append_signup_row(
                    conv_id, lang_code,
                    first_name.strip(), last_name.strip(),
                    program, email.strip(),
                    (phone or "").strip(), consent,
                )
                if ok:
                    st.session_state.signed_up    = True
                    st.session_state.user_first   = first_name.strip()
                    st.session_state.user_last    = last_name.strip()
                    st.session_state.user_email   = email.strip()
                    st.session_state.user_phone   = (phone or "").strip()
                    st.session_state.user_program = program
                    st.success(lbl["ok"])
                    st.rerun()
                else:
                    st.error(lbl["err_write"])  # 具体错误已在 append_signup_row 打日志
    st.stop()


# =========================
# 英文查询翻译为中文
# =========================
def _translate(text: str, instruction: str) -> str:
    resp = get_openai_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def translate_to_chinese(query: str) -> str:
    try:
        return _translate(query, "Translate the following question into Traditional Chinese. Output only the translation, nothing else.")
    except Exception:
        return query


# =========================
# GPT 生成答案
# =========================
WEB_K = 5               # 每个问题向量从官网索引取几段
QA_K = 4                # 每个问题向量从 qa_bank 取几条
MAX_WEB_CHUNKS = 5      # 去重后最多送给 GPT 的官网段落
MAX_QA_CHUNKS = 4       # 去重后最多送给 GPT 的 qa_bank 条目

STRICT_QA_RULES = f"""ANSWERING RULES:
The context is a list of official Q&A pairs maintained by CWTS admissions staff.
1. First decide whether one of the Q&A pairs asks the SAME thing as the question. Details matter:
   a different degree program (e.g. ThM vs MDiv), a different group of students (e.g. local vs international),
   or a different object (e.g. dogs vs cats) means it is NOT the same question.
2. If one pair is the same question, answer using that pair's answer faithfully. Do not add information that is not in it.
3. If no pair is the same question, reply with exactly {UNANSWERABLE_MARKER} and nothing else."""


def stream_answer(query: str, context_chunks: list[dict], strict: bool = False):
    """
    流式生成答案，yield 文字片段。
    strict=True：context 是 qa_bank 的候选问答，只有确实是同一个问题时才作答，否则回 UNANSWERABLE。
    """
    context = "\n\n".join([c["text"] for c in context_chunks])
    lang_name = {"zh": "Simplified Chinese", "zh-TW": "Traditional Chinese", "en": "English"}[lang_code]

    system_prompt = f"""You are an admissions FAQ assistant for Christian Witness Theological Seminary (CWTS).

You can answer questions about:
- Admissions requirements and application procedures
- Degree programs, diplomas, and certificates
- Tuition, scholarships, and financial aid
- Student life, dormitory, and campus
- Faculty members — their background, research interests, and teaching areas
- Accreditation and international student (F-1 visa / I-20) information

Always respond in {lang_name}, regardless of the language of the context provided.
The context may be in Chinese even when the question is in English — this is expected. Read and understand the Chinese context, then answer in {lang_name}.
Be warm, concise, and encouraging.

""" + (STRICT_QA_RULES if strict else f"""ANSWERING RULES:
1. Context items starting with "Q:" / "A:" are official answers written by CWTS admissions staff. When one is relevant, prefer it over website text.
2. If the context contains relevant information, even partial, use it to give the best possible answer.
3. If the question is only partly covered, answer the covered part and suggest contacting admissions@cwts.edu for the rest.
4. Be precise about which degree program or student group a fact applies to; never apply one program's facts to another.
5. If the context does not contain the fact actually asked for (even when it covers the same general topic), reply with {UNANSWERABLE_MARKER} (alone, nothing else). Never write an answer saying the information is not mentioned / not provided.
6. Do NOT guess or make up information not present in the context.""")

    last_line = (
        f'Answer in {lang_name} only if one Q&A pair is the same question; otherwise reply with exactly "{UNANSWERABLE_MARKER}".'
        if strict else
        f'Answer in {lang_name}. If the context does not contain the fact asked for, reply with exactly "{UNANSWERABLE_MARKER}".'
    )
    user_prompt = f"""Question: {query}

Relevant context:
{context}

{last_line}"""

    stream = get_openai_client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=GPT_TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


def query_variants(query: str, zh_query: str | None) -> list[str]:
    """用于检索的问题写法：原文 + 中文翻译（英文提问）/ 简体写法（繁体提问，qa_bank 是简体）"""
    variants = [query]
    if zh_query:
        variants.append(zh_query)
    if lang_code == "zh-TW":
        variants.append(_convert(query, "t2s"))
    return list(dict.fromkeys(v for v in variants if v.strip()))


def _search(load_store, vectors: list[list[float]], k: int) -> list[dict]:
    try:
        store = load_store()
    except Exception as e:
        print(f"[向量库加载失败] {type(e).__name__}: {e}")
        return []
    out = []
    for v in vectors:
        out += [{"text": d.page_content, "source": d.metadata.get("source", "")}
                for d in store.similarity_search_by_vector(v, k=k)]
    return out


def _dedupe(hits: list[dict], limit: int) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        key = h["text"][:100]
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out[:limit]


def retrieve_context(query: str, zh_query: str | None) -> tuple[list[dict], str]:
    """
    向量检索 qa_bank + 官网内容。所有问题写法一次性算向量（只调一次 embedding API），
    再用同一组向量查两个库。qa_bank 在前（staff 审核过的答案优先）。
    返回 (hits, error)：error 非空表示 OpenAI 出错，供 staff 在 unanswered 里查看。
    """
    variants = query_variants(query, zh_query)
    try:
        vectors = get_embeddings().embed_documents(variants)
    except Exception as e:
        print(f"[问题向量计算失败] {type(e).__name__}: {e}")
        return [], f"[系统错误，未能检索资料] {type(e).__name__}"
    qa_hits = _dedupe(_search(load_qa_store, vectors, QA_K), MAX_QA_CHUNKS)
    web_hits = _dedupe(_search(load_faiss_store, vectors, WEB_K), MAX_WEB_CHUNKS)
    return qa_hits + web_hits, ""


def render_answer(query: str, chunks: list[dict], strict: bool = False) -> tuple[str | None, str]:
    """
    流式显示 GPT 的回答，结束后清空占位（最终答案由调用方用 st.success 显示）。
    返回 (answer, error)：answer 为 None 表示无法回答；error 非空表示 OpenAI 出错，供 staff 在 unanswered 里查看。
    """
    placeholder = st.empty()
    placeholder.markdown(t("thinking"))  # 第一个字出来之前的提示
    answer = ""
    try:
        for chunk in stream_answer(query, chunks, strict):
            answer += chunk
            # 可能是 UNANSWERABLE 的开头时先不显示，避免把标记打到屏幕上
            if not _may_be_unanswerable(answer):
                placeholder.markdown(answer + "▌")
    except Exception as e:
        print(f"[OpenAI 生成失败] {type(e).__name__}: {e}")
        placeholder.empty()
        return None, f"[系统错误，未能生成回答] {type(e).__name__}"
    placeholder.empty()
    if not answer.strip() or is_unanswerable(answer):
        return None, ""
    return answer, ""


# =========================
# 主界面
# =========================
st.image("logo.png", width=250)
st.markdown(f"""
<div style='text-align:center;margin-top:15px;margin-bottom:20px;'>
  <h1 style='font-size:34px;font-weight:600;margin-bottom:0;'>{t('title')}</h1>
</div>
""", unsafe_allow_html=True)

# 注册门控
show_signup_form()

# 后台自动重建（每24小时触发一次，不阻塞用户）
trigger_rebuild_if_needed()

# 搜索框 + 提交按钮
st.markdown(f"<h4 style='font-weight:400;margin-top:20px;'><strong>{t('search_prompt')}</strong></h4>",
            unsafe_allow_html=True)

# 用 chat_input：按 Enter 直接发送（Shift+Enter 换行，中文输入法选字的 Enter 不会误发），
# 框内自带发送按钮，发送后自动清空，方便接着问下一个问题。
# 放在 container 里，输入框就留在这里，不会固定到页面底部。
with st.container():
    query = st.chat_input(t("input_placeholder"), max_chars=500)

# =========================
# 检索 + 回答
# =========================
if query:
    conv_id  = get_or_create_conversation_id()
    first    = st.session_state.get("user_first", "")
    last     = st.session_state.get("user_last", "")
    email    = st.session_state.get("user_email", "")
    phone    = st.session_state.get("user_phone", "")
    program  = st.session_state.get("user_program", "")

    # 输入框发送后会清空，所以把刚问的问题显示在回答上方
    st.markdown(f"**{t('your_question')}** {escape_markdown(query)}")
    st.markdown(t("answer_title"))

    # 英文查询先翻译成中文，用于 fuzzy 匹配和向量检索（qa_bank 与官网内容都是中文）
    with st.spinner(t("searching")):
        zh_query = translate_to_chinese(query) if lang_code == "en" else None
        candidates = fuzzy_candidates_qa(zh_query or query)

    answer, error, hits = None, "", []

    # 第一步：qa_bank 里字面相近的候选交给 GPT 核对，确实是同一个问题才作答。
    # 只调一次 GPT（流式，首字很快出现），不做 embedding 检索。
    if candidates:
        qa_chunks = [{"text": f"Q: {c['question']}\nA: {c['answer']}", "source": "qa_bank"}
                     for c in candidates]
        answer, error = render_answer(query, qa_chunks, strict=True)
        if answer:
            hits = qa_chunks
        elif error and lang_code != "en" and is_near_exact(query, candidates[0]["question"]):
            # 降级：GPT 不可用（限流/余额不足等），但问题与 qa_bank 几乎一字不差，
            # 直接给出 staff 写好的答案（按所选简繁转换），不让学生空手而归
            print(f"[降级] GPT 不可用，直接使用 qa_bank 近乎完全匹配的答案：{candidates[0]['question']}")
            answer = _convert(candidates[0]["answer"], "t2s" if lang_code == "zh" else "s2t")
            hits, error = qa_chunks[:1], ""

    # 第二步：没有确认命中（且 OpenAI 没出错）→ 向量检索 + GPT 生成
    if answer is None and not error:
        with st.spinner(t("searching")):
            hits, error = retrieve_context(query, zh_query)
        if hits:
            answer, error = render_answer(query, hits)

    if answer:
        st.success(answer)
        append_pending_row(conv_id, lang_code, first, last, email, program, query, answer,
                           phone_or_weixin=phone)

        with st.expander(t("source_expander"), expanded=False):
            for i, h in enumerate(hits, 1):
                src = f"[{h['source']}]({h['source']})" if h["source"].startswith("http") else h["source"]
                st.markdown(f"**{i}.** {escape_markdown(h['text'][:300])}…\n\n<sub>{t('source_label')}: {src}</sub>",
                            unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.info(t("no_result"))
        append_unanswered_row(
            conv_id, lang_code, first, last,
            email, phone, program, query, error,
        )
