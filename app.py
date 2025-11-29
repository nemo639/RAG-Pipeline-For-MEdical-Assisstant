import streamlit as st
import os
import shutil

# LangChain + HuggingFace
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# HF model loader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------
# AUTO-FIX FAISS folder for uploaded single files
# ---------------------------------------------------------
def ensure_faiss_folder():
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")

    # user uploaded individually, so move them inside folder
    if os.path.exists("index.faiss"):
        shutil.move("index.faiss", "faiss_index/index.faiss")

    if os.path.exists("index.pkl"):
        shutil.move("index.pkl", "faiss_index/index.pkl")

ensure_faiss_folder()

# ---------------------------------------------------------
# Load Embeddings (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ---------------------------------------------------------
# Load FAISS Vector Store (cached)
# FIX: prefix parameter with underscore to avoid hashing error
# ---------------------------------------------------------
@st.cache_resource
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        "faiss_index",
        _embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------------------------------------------------
# Load TinyLlama LLM (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
    )

    return HuggingFacePipeline(pipeline=text_gen)

# ---------------------------------------------------------
# Build RetrievalQA RAG Chain
# ---------------------------------------------------------
def build_qa_chain(llm, vectorstore):
    prompt = PromptTemplate(
        template="""Use ONLY the context below to answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ---------------------------------------------------------
# STREAMLIT APP UI
# ---------------------------------------------------------
st.title("🧠 Medical RAG Assistant")
st.write("Query clinical notes and diagnostic knowledge using RAG.")

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm()
qa_chain = build_qa_chain(llm, vectorstore)

query = st.text_input("Enter your question:")

if query:
    st.write("⏳ Searching…")
    result = qa_chain.invoke({"query": query})

    st.subheader("🟦 Answer")
    st.write(result["result"])

    st.subheader("📄 Source Documents")
    for doc in result["source_documents"]:
        st.markdown(f"**Source ID:** {doc.metadata.get('note_id', 'N/A')}")
        st.write(doc.page_content[:300] + "…")
