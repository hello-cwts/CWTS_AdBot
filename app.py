@st.cache_resource(ttl=3600)   # 每小时自动刷新，防止缓存旧索引
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


@st.cache_resource(ttl=3600)   # 每小时自动刷新，防止缓存旧索引
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