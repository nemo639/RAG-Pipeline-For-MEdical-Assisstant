import streamlit as st
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------
# Auto-fix FAISS folder
# ---------------------------------------------------------
def ensure_faiss_folder():
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")

    if os.path.exists("index.faiss"):
        shutil.move("index.faiss", "faiss_index/index.faiss")

    if os.path.exists("index.pkl"):
        shutil.move("index.pkl", "faiss_index/index.pkl")

ensure_faiss_folder()

# ---------------------------------------------------------
# Load Embeddings
# ---------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ---------------------------------------------------------
# Load FAISS
# ---------------------------------------------------------
@st.cache_resource
def load_vectorstore(embeddings):
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------------------------------------------------
# Load TinyLlama
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7
    )

    return HuggingFacePipeline(pipeline=pipe)

# ---------------------------------------------------------
# Build RAG QA Chain
# ---------------------------------------------------------
def build_qa(llm, vectorstore):
    prompt = PromptTemplate(
        template="""Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("🧠 Medical RAG Assistant")

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm()
qa_chain = build_qa(llm, vectorstore)

query = st.text_input("Ask a question:")

if query:
    result = qa_chain.invoke({"query": query})

    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Sources")
    for doc in result["source_documents"]:
        st.write(doc.page_content[:200] + "...")
