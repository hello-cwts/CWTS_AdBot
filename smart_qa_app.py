import streamlit as st
import time
import re
from datetime import datetime
import json
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai

# 新增：混合检索用
from rapidfuzz import process, fuzz
import numpy as np

# =========================
# Page & Sidebar
# =========================
st.set_page_config(page_title="CWTS - 申请时常见问题答疑", layout="wide")

with st.sidebar:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("### 🌍 語言 / Language")
    lang = st.radio("", ["中文(简)", "中文(繁)", "English"], index=0)
    if lang == "中文(简)":
        lang_code = "zh"
    elif lang == "中文(繁)":
        lang_code = "zh-TW"
    else:
        lang_code = "en"

    st.write("")
    st.markdown(
        """
        <a href="https://www.cwts.edu/zh/admissions/application-procedure/" target="_blank"
           style="display:inline-block;font-size:14px;color:#000;text-decoration:none;border:1px solid #000;
                  padding:6px 12px;border-radius:6px;background-color:#fff;">
           👉 开始申请 / Apply Now
        </a>
        """,
        unsafe_allow_html=True
    )
 
    st.markdown(
        """
        <a href="mailto:admissions@cwts.edu"
           style="display:inline-block;font-size:14px;color:#000;text-decoration:none;border:1px solid #000;
                  padding:6px 12px;border-radius:6px;background-color:#fff;">
           💬 聯絡我們 / Contact us
        </a>
        """,
        unsafe_allow_html=True
    )

# =========================
# Config: your Google Sheets
# =========================
MASTER_SHEET_URL = "https://docs.google.com/spreadsheets/d/1WcNOzUR97NM__k_mFTJrbzhaV18ASYO2cMAZQM7SUx0/edit#gid=1569155313"
QA_SHEET_URL     = "https://docs.google.com/spreadsheets/d/1WcNOzUR97NM__k_mFTJrbzhaV18ASYO2cMAZQM7SUx0/edit#gid=1569155313"

# =========================
# Helpers: Google auth & IO
# =========================
@st.cache_resource
def get_gs_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_SHEET_CREDS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    return gspread.authorize(creds)

def append_signup_row(gs_client, sheet_url, row):
    sh = gs_client.open_by_url(sheet_url)
    try:
        try:
            ws = sh.worksheet("signups")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="signups", rows=2000, cols=20)
            ws.append_row([
                "timestamp","lang","first_name","last_name","program",
                "email","phone_or_weixin","consent"
            ])
        ws.append_row(row)
        return True, ""
    except Exception as e:
        return False, str(e)

@st.cache_data
def load_qa_from_google_sheet():
    gs = get_gs_client()
    sheet = gs.open_by_url(QA_SHEET_URL).sheet1
    return pd.DataFrame(sheet.get_all_records())

@st.cache_resource
def load_faiss_index():
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever()

# =========================
# Verse animation (3s then remove)
# =========================
if "verse_displayed" not in st.session_state:
    st.session_state.verse_displayed = False

quote_area = st.empty()
if not st.session_state.verse_displayed:
    quote_area.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC&family=Playfair+Display&display=swap');
    @keyframes fadeInOut {{ 0% {{opacity:0;}} 15% {{opacity:1;}} 85% {{opacity:1;}} 100% {{opacity:0;}} }}
    .fade-in-out {{ animation: fadeInOut 3s ease-in-out forwards; }}
    .bible-verse-box {{ background: linear-gradient(to bottom right, rgba(255,255,255,.8), rgba(240,240,240,.85));
        padding:30px 40px; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,.1); display:inline-block; }}
    </style>
    <div class="fade-in-out" style="text-align:center; margin-top:60px; margin-bottom:60px;">
      <div class="bible-verse-box" style="font-family:'Noto Serif SC', serif; font-size:20px; line-height:2; color:#444;">
        <div style="margin-bottom:18px;">
            我又聽見主的聲音說：<br>
            「我可以差遣誰呢？誰肯為我們去呢？」<br>
            我說：「我在這裡，請差遣我！」<br>
            —— 《以賽亞書》6:8
        </div>
        <div style="font-family:'Playfair Display', serif; font-size:16px; color:#666;">
            Then I heard the voice of the Lord saying,<br>
            “Whom shall I send? And who will go for us?”<br>
            And I said, “Here am I. Send me!”<br>
            — Isaiah 6:8
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(4)
    quote_area.empty()
    st.session_state.verse_displayed = True

# =========================
# Signup form (must pass before Q&A)
# =========================
def show_signup_form(lang_code: str, sheet_url_master: str):
    labels = {
        "zh": {
            "title": "📝 基本信息 / Basic Info",
            "desc":  "为了更好地协助您，请先填写以下信息（仅用于学院招生与服务）。",
            "first": "名",
            "last":  "姓",
            "program":"目标学位项目（请点击选择）",
            "email": "电子邮件",
            "phone": "电话或微信（选填）",
            "consent":"我同意学院以此资料联系我",
            "submit":"提交",
            "err_name":"请输入姓名",
            "err_email":"请输入并完成表格",
            "err_consent":"请勾选同意条款",
            "ok":"已收到，感谢！您现在可以提问啦。",
        },
        "zh-TW": {
            "title": "📝 基本資訊 / Basic Info",
            "desc":  "為了更好地協助您，請先填寫以下資訊（僅用於學院招生與服務）。",
            "first": "名",
            "last":  "姓",
            "program": "目標學位項目（請點擊選擇）",
            "email": "電子郵件",
            "phone": "電話或微信（選填）",
            "consent":"我同意學院以此資料聯絡我",
            "submit":"提交",
            "err_name":"請輸入姓名 / Please enter your name",
            "err_email":"請輸入並完成表格",
            "err_consent":"請勾選同意條款 / Please provide consent",
            "ok":"已收到，感謝！您現在可以提問囉。",
        },
        "en": {
            "title": "📝 Basic Info",
            "desc":  "To serve you better, please fill out the form below (used for admissions & support only).",
            "first": "First name",
            "last":  "Last name",
            "program": "Target Degree Program (Please Click to Select)",     
            "email": "Email",
            "phone": "Phone or Weixin (optional)",
            "consent":"I consent to be contacted by the seminary",
            "submit":"Submit",
            "err_name":"Please enter your name",
            "err_email":"Please enter a valid email",
            "err_consent":"Please provide consent",
            "ok":"Thank you! You can ask questions now.",
        }
    }

    program_options = {
        "zh": [
            "教牧博士（DMin）",
            "神学硕士（ThM）",
            "道学硕士（MDiv）",
            "圣经研究硕士（MBS）",
            "基督教圣工硕士（MCM）",
            "基督教圣工硕士（网络）（MCM Online）",
            "神学精要硕士（MTE）",
            "基督教研究文凭（DCS）",
            "教会事工文凭（DCL）",
            "基督工人证书",
            "家庭事工证书",
            "儿童事工证书",
            "宣教事工证书",
            "其他 / Other"
        ],
        "zh-TW": [
            "教牧博士（DMin）",
            "神學碩士（ThM）",
            "道學碩士（MDiv）",
            "聖經研究碩士（MBS）",
            "基督教聖工碩士（MCM）",
            "基督教聖工碩士（網路）（MCM Online）",
            "神學精要碩士（MTE）",
            "基督教研究文憑（DCS）",
            "教會事工文憑（DCL）",
            "基督工人證書",
            "家庭事工證書",
            "兒童事工證書",
            "宣教事工證書",
            "其他 / Other"
        ],
        "en": [
            "Doctor of Ministry (DMin)",
            "Master of Theology (ThM)",
            "Master of Divinity (MDiv)",
            "Master of Biblical Studies (MBS)",
            "Master of Christian Ministry (MCM)",
            "Master of Christian Ministry (Online) (MCM Online)",
            "Master of Theological Essentials (MTE)",
            "Diploma of Christian Studies (DCS)",
            "Diploma in Church Leadership (DCL)",
            "Certificate in Servant Leadership / Certificate of Sunday School Teacher / Certificate of Small Group Leader",
            "Certificate of Family Ministry / Certificate of Family Ministry Teacher",
            "Certificate of Children Ministry",
            "Certificate of Evangelism Ministry / Certificate of Evangelism Ministry Teacher",
            "Other"
        ]
    }

    if "signed_up" not in st.session_state:
        st.session_state.signed_up = False
    if st.session_state.signed_up:
        return

    st.markdown(f"### {labels[lang_code]['title']}")
    st.caption(labels[lang_code]["desc"])

    # （可选）把名/姓行距压缩一点
    st.markdown("""
    <style>
      .signup-field { margin-top: 8px; }
      div.stForm button[kind="formSubmit"]{ width:100%; border-radius:8px; padding:.6rem 0; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

    with st.form("signup_form", clear_on_submit=False):
        # —— 名 / 姓：同一行两个 columns ——
        col1, col2 = st.columns(2)

        with col1:
            first_name = st.text_input(
                label=labels[lang_code]["first"],
                placeholder="e.g., Zach/一凡",
                max_chars=60
            )

        with col2:
            last_name = st.text_input(
                label=labels[lang_code]["last"],
                placeholder="e.g., Wei/魏",
                max_chars=60
            )

        # 其余字段单列即可
        email = st.text_input(labels[lang_code]["email"],
                              key="email_input",
                              label_visibility="visible",
                              max_chars=120, placeholder="name@example.com")

        phone = st.text_input(labels[lang_code]["phone"],
                              key="phone_input",
                              label_visibility="visible",
                              max_chars=60, placeholder="+1 650-123-4567 / weixin-id")

        program = st.selectbox(labels[lang_code]["program"],
                               program_options[lang_code],
                               index=0)

        consent = st.checkbox(labels[lang_code]["consent"])

        submitted = st.form_submit_button(labels[lang_code]["submit"])

        if submitted:
            # 校验
            email_ok = re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or "")
            if not (first_name and last_name):
                st.error(labels[lang_code]["err_name"])
            elif not email_ok:
                st.error(labels[lang_code]["err_email"])
            elif not consent:
                st.error(labels[lang_code]["err_consent"])
            else:
                gs_client = get_gs_client()
                ok, err = append_signup_row(
                    gs_client,
                    sheet_url_master,
                    [
                        datetime.utcnow().isoformat(),
                        lang_code,
                        first_name.strip(),
                        last_name.strip(),
                        program,
                        email.strip(),
                        (phone or "").strip(),
                        "yes" if consent else "no",
                    ],
                )
                if ok:
                    st.success(labels[lang_code]["ok"])
                    st.session_state.signed_up = True
                    st.rerun()
                else:
                    st.error(f"寫入失敗 / Failed to write: {err}")

    # 未通过就阻止下方渲染
    st.stop()

# ---- 必须在搜索与 Q&A 前调用 ----
show_signup_form(lang_code, MASTER_SHEET_URL)

# =========================
# Data & Retriever
# =========================
retriever = load_faiss_index()
df = load_qa_from_google_sheet()

# =========================
# Hybrid search helpers
# =========================
@st.cache_data(show_spinner=False)
def build_question_lists(df_all: pd.DataFrame):
    """
    为模糊匹配做语言分桶，缓存。
    """
    buckets = {}
    for code in df_all["lang"].dropna().unique():
        sub = df_all[df_all["lang"] == code].copy()
        buckets[code] = {
            "questions": sub["question"].fillna("").tolist(),
            "rows": sub.reset_index(drop=True),
        }
    return buckets

def hybrid_search(query, lang_code, retriever, df_all, topk=6):
    """
    关键词子串 + 模糊匹配 + 语义检索；返回若干条用于上下文的文本。
    """
    out = []
    q = (query or "").strip()
    if not q:
        return out

    # 1) 关键词子串
    sub = df_all[df_all["lang"] == lang_code].copy()
    if not sub.empty:
        mask = (
            sub["question"].fillna("").str.contains(q, case=False, regex=False) |
            sub["answer"].fillna("").str.contains(q, case=False, regex=False)
        )
        kw_hits = sub[mask].head(topk)
        for _, r in kw_hits.iterrows():
            out.append({
                "text": f"Q: {r['question']}\nA: {r['answer']}",
                "source": "keyword",
                "score": 1.0
            })

    # 2) 模糊匹配
    buckets = build_question_lists(df_all)
    bucket = buckets.get(lang_code)
    if bucket and bucket["questions"]:
        matches = process.extract(
            q, bucket["questions"],
            scorer=fuzz.WRatio,
            processor=None,
            limit=topk
        )
        for mtext, mscore, midx in matches:
            row = bucket["rows"].iloc[midx]
            out.append({
                "text": f"Q: {row['question']}\nA: {row['answer']}",
                "source": "fuzzy",
                "score": float(mscore) / 100.0
            })

    # 3) 语义检索
    try:
        sem_docs = retriever.get_relevant_documents(q)
        for d in sem_docs[:topk]:
            out.append({
                "text": d.page_content,
                "source": "semantic",
                "score": 0.9
            })
    except Exception:
        pass

    # 4) 去重 + 排序
    seen = set()
    deduped = []
    for item in sorted(out, key=lambda x: x["score"], reverse=True):
        key = item["text"].strip()
        if key not in seen:
            deduped.append(item)
            seen.add(key)

    return deduped[:topk]

# =========================
# Title & Search prompt (dynamic by lang)
# =========================
st.image("logo.png", width=250)

titles = {
    "zh": "CWTS - 申请常见问题答疑",
    "zh-TW": "CWTS - 申請常見問題答疑",
    "en": "CWTS - Application Frequently Asked Questions"
}
st.markdown(f"""
<div style='text-align:center; margin-top:15px; margin-bottom:20px;'>
  <h1 style='font-size:34px; font-weight:600; margin-bottom:0;'>{titles[lang_code]}</h1>
</div>
""", unsafe_allow_html=True)

search_prompts = {
    "zh": "🔍 请输入您的问题（例如：如何申请奖学金/什么是财务证明）",
    "zh-TW": "🔍 請輸入您的問題（例如：如何申請獎學金/什麼是財務證明）",
    "en": "🔍 Please enter your question (e.g., How to apply for a scholarship / What is a financial statement?)"
}
st.markdown(f"""
<h4 style='font-size:22px; font-weight:400; margin-top:20px;'><strong>{search_prompts[lang_code]}</strong></h4>
""", unsafe_allow_html=True)

query = st.text_input(
    label="query_input",
    placeholder={
        "zh": "請在此輸入問題……（支持简体 / 繁體 / English）",
        "zh-TW": "請在此輸入問題……（支援簡體 / 繁體 / English）",
        "en": "Enter your question here… (Chinese/English supported)"
    }[lang_code],
    label_visibility="collapsed"
)

if query:
    hits = hybrid_search(query, lang_code, retriever, df, topk=6)

    if hits:
        # 拼接上下文
        context = "\n\n".join([h["text"] for h in hits[:4]])

        prompt = f"""
You are an admissions FAQ assistant for Christian Witness Theological Seminary (CWTS) only can answer the question about this school and admission.
Answer briefly, clearly, and warmly, in the same language（Simplified Chinese/Traditional Chinese/English as the user's question.
Encourage the user to complete the application if appropriate. 

Question:
{query}

Relevant context (may contain Q&A snippets):
{context}

Now write the answer in the user's language:
        """.strip()

        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        # 先显示回答标题
        st.markdown("### 🙋 回答 / Answer")

        # 加载动画
        with st.spinner("⏳ 正在生成回答，请稍候..."):
            resp = (client.chat_completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.3,
                    ) if hasattr(client, "chat_completions")
                   else client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.3,
                   ))

        try:
            ai_answer = resp.choices[0].message.content.strip()
        except:
            ai_answer = (resp.choices[0].delta.content.strip()
                         if hasattr(resp.choices[0], "delta") and resp.choices[0].delta.content else "")

        st.success(ai_answer or {
            "zh": "（已生成回答，但解析返回时为空，请稍后重试）",
            "zh-TW": "（已生成回答，但解析返回時為空，請稍後重試）",
            "en": "(Answer generated, but response body was empty; please try again.)"
        }[lang_code])

        # 再显示命中条目
        with st.expander({
            "zh": "🔎 命中的相关条目（点击展开）",
            "zh-TW": "🔎 命中的相關條目（點此展開）",
            "en": "🔎 Matched items (click to expand)"
        }[lang_code], expanded=False):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**{i}.** {h['text']}\n\n<sub>source: {h['source']}</sub>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

    else:
        st.info({
            "zh": "未找到相關內容，請嘗試換個說法或輸入更完整的關鍵字～",
            "zh-TW": "未找到相關內容，請嘗試換個說法或輸入更完整的關鍵字～",
            "en": "No relevant answer found. Try rephrasing or adding more keywords."
        }[lang_code])


# =========================
# Q&A List
# =========================
st.markdown("---")
qa_titles = {
    "zh": "📋 全部常见问题",
    "zh-TW": "📋 全部常見問題",
    "en": "📋 All Frequently Asked Questions"
}
st.markdown(f"""
<h4 style='font-size:22px; font-weight:400; margin-top:20px;'>
<strong>{qa_titles[lang_code]}</strong>
</h4>
""", unsafe_allow_html=True)

filtered_df = df[df["lang"] == lang_code].reset_index(drop=True)
for i, row in filtered_df.iterrows():
    st.markdown(f"**Q{i+1}: {row['question']}**")
    st.markdown(f"👉 {row['answer']}")
    st.markdown("<hr style='margin-top: 16px; margin-bottom: 24px;'>", unsafe_allow_html=True)
