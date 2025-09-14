import pandas as pd
import openai
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# 加载 OpenAI API Key（从 secrets.toml 中）
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 设置 Google Sheets 授权
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
client = gspread.authorize(creds)

# Google Sheet 链接（请替换为你自己的链接）
sheet_url = st.secrets["GOOGLE_SHEET_ID"]
sheet = client.open_by_url(sheet_url).sheet1

# 获取数据并转为 DataFrame（要求有 'question' 和 'answer' 两列）
data = sheet.get_all_records()
df = pd.DataFrame(data)

# 合并文本用于向量嵌入（可以改为只用 question）
texts = (df["question"] + " " + df["answer"]).tolist()

# 创建嵌入模型并生成向量
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embedding_model)

# 保存 FAISS 索引到本地
vectorstore.save_local("faiss_index")

print("✅ 向量库构建完成，已保存为 'faiss_index'")



