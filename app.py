import streamlit as st
import os
import json
import pandas as pd
import zipfile
import rarfile
from pathlib import Path
import time
import re

# LangChain and ML imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List

# Page configuration
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.all_chunks = []
    st.session_state.df_notes = None
    st.session_state.chat_history = []

# Custom Retriever Class
class WorkingRetriever(BaseRetriever):
    vectorstore: object
    all_chunks: List[Document]
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vectorstore, all_chunks, k=5):
        super().__init__(
            vectorstore=vectorstore,
            all_chunks=all_chunks,
            k=k
        )

    def extract_note_id(self, query):
        pattern = r'\b(\d{8}-[A-Z]{2}-\d+)\b'
        match = re.search(pattern, query)
        return match.group(1) if match else None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        note_id_pattern = self.extract_note_id(query)

        if note_id_pattern:
            matching_chunks = [
                chunk for chunk in self.all_chunks
                if note_id_pattern in chunk.metadata.get('note_id', '')
            ]

            if matching_chunks:
                if len(matching_chunks) > self.k:
                    ranked = self.vectorstore.similarity_search(query, k=50)
                    matching_note_ids = set([c.metadata.get('note_id') for c in matching_chunks])
                    ranked_matches = [
                        doc for doc in ranked
                        if doc.metadata.get('note_id') in matching_note_ids
                    ]
                    if ranked_matches:
                        return ranked_matches[:self.k]

                sorted_matches = sorted(
                    matching_chunks,
                    key=lambda x: x.metadata.get('chunk_index', 0)
                )
                return sorted_matches[:self.k]

        return self.vectorstore.similarity_search(query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# Helper functions
@st.cache_data
def load_clinical_notes(root_dir):
    """Load clinical notes from JSON files"""
    all_notes = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        note_data = json.load(f)

                    complaint = note_data.get('input1', '')
                    hpi = note_data.get('input2', '')
                    pmh = note_data.get('input3', '')
                    fh = note_data.get('input4', '')
                    pe = note_data.get('input5', '')
                    labs = note_data.get('input6', '')

                    full_text = (
                        f"CHIEF COMPLAINT: {complaint}\n"
                        f"HISTORY OF PRESENT ILLNESS: {hpi}\n"
                        f"PAST MEDICAL HISTORY: {pmh}\n"
                        f"FAMILY HISTORY: {fh}\n"
                        f"PHYSICAL EXAMINATION: {pe}\n"
                        f"LABS/IMAGING: {labs}"
                    )

                    diag_key = [k for k in note_data if not k.startswith("input")]
                    diagnosis = diag_key[0].split("$")[0] if diag_key else "Unknown"

                    all_notes.append({
                        'note_id': os.path.basename(fpath),
                        'full_text': full_text,
                        'diagnosis': diagnosis
                    })
                except:
                    continue

    return pd.DataFrame(all_notes)

def create_chunks(df_notes, text_splitter):
    """Create text chunks from clinical notes"""
    all_chunks = []
    
    for idx, row in df_notes.iterrows():
        note_id = row['note_id']
        diagnosis = row.get('diagnosis', 'Unknown')
        full_text = row['full_text']

        enhanced_text = f"NOTE_ID: {note_id}\nDIAGNOSIS: {diagnosis}\n\n{full_text}"

        chunks = text_splitter.create_documents(
            texts=[enhanced_text],
            metadatas=[{
                'note_id': note_id,
                'diagnosis': diagnosis,
                'type': 'clinical_note'
            }]
        )

        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i

        all_chunks.extend(chunks)

    return all_chunks

@st.cache_resource
def load_models():
    """Load embedding model and LLM"""
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load LLM
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    
    return embeddings, llm

def initialize_system(data_path):
    """Initialize the RAG system"""
    try:
        with st.spinner("Loading clinical notes..."):
            df_notes = load_clinical_notes(data_path)
            st.session_state.df_notes = df_notes
        
        st.success(f"✓ Loaded {len(df_notes)} clinical notes")
        
        with st.spinner("Creating text chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " "],
                length_function=len
            )
            all_chunks = create_chunks(df_notes, text_splitter)
            st.session_state.all_chunks = all_chunks
        
        st.success(f"✓ Created {len(all_chunks)} text chunks")
        
        with st.spinner("Loading AI models (this may take a few minutes)..."):
            embeddings, llm = load_models()
        
        st.success("✓ Models loaded successfully")
        
        with st.spinner("Building vector database..."):
            vectorstore = FAISS.from_documents(
                documents=all_chunks,
                embedding=embeddings
            )
            st.session_state.vectorstore = vectorstore
        
        st.success(f"✓ Vector store created with {vectorstore.index.ntotal} vectors")
        
        with st.spinner("Setting up QA chain..."):
            prompt_template = """You are a clinical assistant. Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            retriever = WorkingRetriever(
                vectorstore=vectorstore,
                all_chunks=all_chunks,
                k=5
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            st.session_state.qa_chain = qa_chain
        
        st.success("✓ QA system ready!")
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return False

# Main UI
st.markdown('<h1 class="main-header">🏥 Clinical RAG System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    data_path = st.text_input(
        "Data Directory Path",
        value="./DiReCT_Notes_Sample/Finished",
        help="Path to the directory containing clinical note JSON files"
    )
    
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        if os.path.exists(data_path):
            initialize_system(data_path)
        else:
            st.error(f"Path not found: {data_path}")
    
    st.markdown("---")
    
    if st.session_state.initialized:
        st.success("✓ System Ready")
        
        st.metric("Clinical Notes", len(st.session_state.df_notes))
        st.metric("Text Chunks", len(st.session_state.all_chunks))
        st.metric("Vectors", st.session_state.vectorstore.index.ntotal)
    else:
        st.warning("⚠️ System Not Initialized")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content area
if not st.session_state.initialized:
    st.info("👈 Please initialize the system using the sidebar")
    
    with st.expander("📖 About This System"):
        st.markdown("""
        ### Clinical RAG System
        
        This system uses **Retrieval-Augmented Generation (RAG)** to answer questions about clinical notes.
        
        **How it works:**
        1. **Retrieval**: Searches through clinical notes to find relevant information
        2. **Augmentation**: Provides context to the AI model
        3. **Generation**: Creates accurate answers based on retrieved data
        
        **Features:**
        - Search by patient ID (e.g., "patient 18427803-DS-5")
        - General medical queries
        - Source document citations
        - Grounded responses (no hallucinations)
        """)

else:
    # Chat interface
    st.subheader("💬 Ask Questions About Clinical Notes")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('note_id', 'Unknown')}")
                        st.text(doc.page_content[:200] + "...")
    
    # Chat input
    if query := st.chat_input("Ask a question about the clinical notes..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": query})
                answer = result["result"].strip()
                sources = result["source_documents"]
                
                # Clean answer
                if "ANSWER:" in answer:
                    answer = answer.split("ANSWER:")[-1].strip()
                
                st.markdown(answer)
                
                # Show sources
                with st.expander("📄 View Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('note_id', 'Unknown')}")
                        st.text(doc.page_content[:200] + "...")
                
                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Clinical RAG System | Powered by LangChain & Streamlit</div>",
    unsafe_allow_html=True
)
