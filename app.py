"""
app.py
CWTS 招生FAQ Chatbot v2
"""

import time
import streamlit as st
import re
from datetime import datetime
from pathlib import Path

from config import (
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    GPT_MODEL,
    GPT_TEMPERATURE,
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
        "zh": "在此输入问题…按回车提交",
        "zh-TW": "在此輸入問題…按 Enter 提交",
        "en": "Type your question here… press Enter to submit",
    },
    "answer_title": {
        "zh": "### 🙋 回答",
        "zh-TW": "### 🙋 回答",
        "en": "### 🙋 Answer",
    },
    "thinking": {
        "zh": "⏳ 正在生成回答，请稍候...",
        "zh-TW": "⏳ 正在生成回答，請稍候...",
        "en": "⏳ Generating answer, please wait...",
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
# FAISS 检索器
# =========================
@st.cache_resource(ttl=3600)
def load_faiss_retriever():
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model=EMBEDDING_MODEL,
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True,
    ).as_retriever(search_kwargs={"k": 5})


@st.cache_resource(ttl=3600)
def load_qa_faiss_retriever():
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model=EMBEDDING_MODEL,
    )
    return FAISS.load_local(
        "faiss_index_qa", embeddings,
        allow_dangerous_deserialization=True,
    ).as_retriever(search_kwargs={"k": 3})


# =========================
# 重建向量库（后台）
# =========================
REBUILD_INTERVAL_HOURS = 24
LAST_REBUILD_FILE = "last_rebuild.txt"

def should_rebuild() -> bool:
    try:
        last = float(Path(LAST_REBUILD_FILE).read_text().strip())
        elapsed = (datetime.utcnow().timestamp() - last) / 3600
        return elapsed >= REBUILD_INTERVAL_HOURS
    except Exception:
        return True

def mark_rebuilt():
    Path(LAST_REBUILD_FILE).write_text(str(datetime.utcnow().timestamp()))

def async_rebuild():
    try:
        from scraper import rebuild_index
        rebuild_index(st.secrets["OPENAI_API_KEY"])
        mark_rebuilt()
        load_faiss_retriever.clear()
        load_qa_faiss_retriever.clear()
    except Exception as e:
        print(f"[后台重建失败] {e}")


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
.verse-wrap{display:flex;justify-content:center;align-items:center;margin:56px 0}
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
<br/><br/>
<div class="verse-wrap fade-in-out">
  <div class="verse-card">
    <div class="badge">校訓</div>
    <div class="verse-text">立於聖言勤於禱<br/>終於使命敏於行</div>
    <div class="verse-sep"></div>
  </div>
</div>
""", unsafe_allow_html=True)
    time.sleep(4)
    quote_area.empty()
    st.session_state.verse_displayed = True


# =========================
# 注册表单（必须通过才能提问）
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
              "err_consent":"请勾选同意条款","ok":"已收到，感谢！您现在可以提问啦。"},
    "zh-TW": {"title":"📝 基本資訊","desc":"為了更好地協助您，請先填寫以下資訊。",
              "first":"名*","last":"姓*","program":"目標學位項目*",
              "email":"電子郵件*","phone":"電話或微信（選填）",
              "consent":"我同意學院以此資料聯絡我*","submit":"提交",
              "err_name":"請輸入姓名","err_email":"請輸入正確的郵箱",
              "err_consent":"請勾選同意條款","ok":"已收到，感謝！您現在可以提問囉。"},
    "en":    {"title":"📝 Basic Info","desc":"To serve you better, please fill out the form below.",
              "first":"First name*","last":"Last name*","program":"Target Degree Program*",
              "email":"Email*","phone":"Phone or Weixin (optional)",
              "consent":"I consent to be contacted by the seminary*","submit":"Submit",
              "err_name":"Please enter your name","err_email":"Please enter a valid email",
              "err_consent":"Please provide consent","ok":"Thank you! You can ask questions now."},
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
                    st.session_state.user_name    = f"{first_name.strip()} {last_name.strip()}"
                    st.session_state.user_email   = email.strip()
                    st.session_state.user_phone   = (phone or "").strip()
                    st.session_state.user_program = program
                    st.success(lbl["ok"])
                    st.rerun()
                else:
                    st.error(f"写入失败：{err}")
    st.stop()


# =========================
# 检索函数
# =========================
def search_faiss(query: str) -> list[dict]:
    try:
        retriever = load_faiss_retriever()
        docs = retriever.invoke(query)
        return [{"text": d.page_content, "source": d.metadata.get("source", "")} for d in docs]
    except Exception:
        return []


def search_qa_bank(query: str) -> list[dict]:
    try:
        retriever = load_qa_faiss_retriever()
        docs = retriever.invoke(query)
        return [{"text": d.page_content, "source": "qa_bank"} for d in docs]
    except Exception:
        return []


# =========================
# GPT 生成答案
# =========================
UNANSWERABLE_MARKER = "UNANSWERABLE"

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    import openai
    context = "\n\n".join([c["text"] for c in context_chunks[:4]])
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
Be warm, concise, and encouraging. When mentioning a faculty member, make it personal and inviting — help the applicant feel connected to the seminary community.

IMPORTANT: If the context does not contain enough information to answer the question confidently,
you MUST reply with exactly this single word and nothing else: {UNANSWERABLE_MARKER}
Do NOT guess, do NOT make up information, do NOT say "the seminary does not offer X" unless the context explicitly states it."""

    user_prompt = f"""Question: {query}

Relevant context:
{context}

If the context does not clearly answer the question, reply with exactly: {UNANSWERABLE_MARKER}"""

    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=GPT_TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()


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

# 搜索框 + 提交按钮
st.markdown(f"<h4 style='font-weight:400;margin-top:20px;'><strong>{t('search_prompt')}</strong></h4>",
            unsafe_allow_html=True)

with st.form("query_form", clear_on_submit=False):
    col1, col2 = st.columns([11, 1])
    with col1:
        query = st.text_input(
            label="query",
            placeholder=t("input_placeholder"),
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("→", use_container_width=True)

if not submitted:
    query = None

# =========================
# 检索 + 回答
# =========================
if query:
    conv_id  = get_or_create_conversation_id()
    name     = st.session_state.get("user_name", "")
    email    = st.session_state.get("user_email", "")
    phone    = st.session_state.get("user_phone", "")
    program  = st.session_state.get("user_program", "")
    first, *rest = name.split() if name else ("", [])
    last = " ".join(rest)

    # 同时搜网站向量库和 qa_bank，合并结果
    hits = search_faiss(query)
    qa_hits = search_qa_bank(query)

    # qa_bank 结果优先放前面（staff维护的答案更可靠）
    combined = qa_hits + hits

    # 去重
    seen = set()
    hits = []
    for h in combined:
        key = h["text"][:100]
        if key not in seen:
            hits.append(h)
            seen.add(key)

    if hits:
        st.markdown(t("answer_title"))
        with st.spinner(t("thinking")):
            answer = generate_answer(query, hits)

        if answer == UNANSWERABLE_MARKER:
            st.info(t("no_result"))
            append_unanswered_row(
                conv_id, lang_code, first, last,
                email, phone, program, query, "",
            )
        else:
            st.success(answer)
            append_pending_row(conv_id, lang_code, first, last, email, program, query, answer)

            with st.expander(t("source_expander"), expanded=False):
                for i, h in enumerate(hits, 1):
                    src = f"[{h['source']}]({h['source']})" if h["source"].startswith("http") else h["source"]
                    st.markdown(f"**{i}.** {h['text'][:300]}…\n\n<sub>来源: {src}</sub>",
                                unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.info(t("no_result"))
        append_unanswered_row(
            conv_id, lang_code, first, last,
            email, phone, program, query, "",
        )