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
# MOVE UPLOADED FAISS FILES INTO A FOLDER
# ---------------------------------------------------------
def ensure_faiss_folder():
    os.makedirs("faiss_index", exist_ok=True)

    if os.path.exists("index.faiss"):
        shutil.move("index.faiss", "faiss_index/index.faiss")

    if os.path.exists("index.pkl"):
        shutil.move("index.pkl", "faiss_index/index.pkl")

ensure_faiss_folder()


# ---------------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------------
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


# ---------------------------------------------------------
# LOAD FAISS DB (NO CACHE)
# ---------------------------------------------------------
def load_vectorstore():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


# ---------------------------------------------------------
# LOAD TINYLLAMA (NO CACHE)
# ---------------------------------------------------------
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
    )

    return HuggingFacePipeline(pipeline=pipe)


# ---------------------------------------------------------
# BUILD RETRIEVAL QA
# ---------------------------------------------------------
def build_qa_chain(llm, vectorstore):

    prompt = PromptTemplate(
        template="""Use ONLY the context below to answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🧠 Medical RAG Assistant")
st.write("Ask a clinical question based on the uploaded FAISS index.")

# Load everything (no cache → no hashing errors)
vectorstore = load_vectorstore()
llm = load_llm()
qa_chain = build_qa_chain(llm, vectorstore)

query = st.text_input("Enter your question:")

if query:
    st.write("⏳ Processing…")
    result = qa_chain.invoke({"query": query})

    st.subheader("🔵 Answer")
    st.write(result["result"])

    st.subheader("📄 Source Documents")
    for doc in result["source_documents"]:
        st.write(f"**ID:** {doc.metadata.get('note_id', 'N/A')}")
        st.write(doc.page_content[:300] + "…")
