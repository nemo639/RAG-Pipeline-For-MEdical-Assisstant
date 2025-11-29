import streamlit as st
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ---------------------------------------------------------
# Auto-fix FAISS folder (if user uploaded files individually)
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
# Build RAG Chain (New LangChain v0.3+ API)
# ---------------------------------------------------------
def build_rag_chain(llm, vectorstore):

    prompt = PromptTemplate(
        template="""Use ONLY the below context to answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
        input_variables=["context", "question"]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("🧠 Medical RAG Assistant")
st.write("Ask clinical questions. Answers come strictly from the FAISS-indexed notes.")

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
llm = load_llm()
rag_chain = build_rag_chain(llm, vectorstore)

query = st.text_input("Enter your question:")

if query:
    response = rag_chain.invoke({"question": query})

    st.subheader("Answer:")
    st.write(response["answer"])

    st.subheader("Sources:")
    for doc in response["context"]:
        st.write(doc.page_content[:200] + "...")
