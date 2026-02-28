# Optimized version
import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from src.document_processor import load_and_process_pdf
from src.embeddings import build_vectorstore, load_vectorstore, INDEX_DIR
from src.rag_pipeline import get_qa_chain, ask

# Load environment variables
load_dotenv()

PDF_PATH = os.path.join("data", "swiggy_annual_report.pdf")

st.set_page_config(page_title="Swiggy Annual Report Q&A", page_icon="🍕", layout="centered")

# Inject Custom CSS for Swiggy Branding
st.markdown("""
    <style>
    /* Global App Background & Text */
    .stApp {
        background-color: #1C1C1C;
        color: #FFFFFF;
    }
    
    /* Remove default top padding */
    .css-18e3th9 {
        padding-top: 0rem;
    }
    .block-container {
        padding-top: 1rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FC8019;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e06b0a;
        color: white;
    }
    
    /* Text Input */
    .stTextInput>div>div>input {
        background-color: #2A2A2A;
        color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #444;
    }
    .stTextInput>div>div>input:focus {
        border-color: #FC8019;
        box-shadow: 0 0 0 1px #FC8019;
    }

    /* Banners & Cards */
    .header-banner {
        background-color: #FC8019;
        padding: 1.5rem;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .header-banner-left {
        display: flex;
        flex-direction: column;
    }
    .header-title {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
        padding: 0;
    }
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        margin: 0;
        padding-top: 0.2rem;
    }
    .badge-dark-pill {
        background-color: #1C1C1C;
        color: #FC8019;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        white-space: nowrap;
    }
    
    /* Answer Card */
    .answer-card {
        background-color: #2A2A2A;
        border-left: 4px solid #FC8019;
        padding: 1.5rem;
        border-radius: 8px;
        color: #FFFFFF;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .answer-card-error {
        border-left: 4px solid #FC5C5C;
    }
    
    /* Pills */
    .pill-green {
        background-color: #48BB78;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .pill-red {
        background-color: #FC5C5C;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .pill-dark-badge {
        background-color: #2A2A2A;
        color: #AAAAAA;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        display: inline-block;
    }

    /* Source Cards */
    .source-card {
        background-color: #2A2A2A;
        border-top: 2px solid rgba(252, 128, 25, 0.5);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .source-title {
        color: #FC8019;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .source-text {
        color: #DDDDDD;
        font-size: 0.9rem;
    }

    /* Labels & Muted Text */
    .input-label {
        color: #FC8019;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .muted-text {
        color: #AAAAAA;
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #1C1C1C;
    }
    .sidebar-title {
        color: #FC8019;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] hr {
        border-color: #444;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🍕 About this App</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #DDDDDD; font-size: 0.9rem; margin-bottom: 1.5rem;">
        <ul>
            <li><b>Retrieval:</b> Scans the entire annual report.</li>
            <li><b>Augmented:</b> Injects relevant facts into the AI prompt.</li>
            <li><b>Generation:</b> AI synthesizes a clean, grounded answer.</li>
        </ul>
    </div>
    <hr style="background-color: #444; height: 1px; border: none; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="pill-dark-badge">Model: LLaMA 3.3 70B via Groq</div>', unsafe_allow_html=True)
    st.markdown('<div class="pill-dark-badge">Embeddings: MiniLM-L3-v2</div>', unsafe_allow_html=True)
    st.markdown('<div class="pill-dark-badge">Vector DB: FAISS</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 2rem; font-size: 0.8rem; color: #AAAAAA; text-align: center;">Built for Swiggy Annual Report Analysis</div>', unsafe_allow_html=True)


# ----------------- MAIN AREA -----------------

# Header Banner
st.markdown("""
<div class="header-banner">
    <div class="header-banner-left">
        <h1 class="header-title">🍕 Swiggy Annual Report</h1>
        <p class="header-subtitle">AI-Powered Q&A — Strictly grounded in the official report</p>
    </div>
    <div class="badge-dark-pill">Powered by RAG</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_qa_chain():
    """
    Initializes the QA chain.
    If the FAISS index doesn't exist, it processes the PDF and builds it.
    If it does, it simply loads the existing index.
    """
    if not os.path.exists(INDEX_DIR):
        if not os.path.exists(PDF_PATH):
            st.error(f"PDF file not found! Please place the Swiggy Annual Report PDF at `{PDF_PATH}`.")
            st.stop()
            
        with st.spinner("Building optimized index, this may take a minute..."):
            chunks = load_and_process_pdf(PDF_PATH)
            vectorstore = build_vectorstore(chunks)
            st.success(f"Optimized index built successfully! Indexed {len(chunks)} unique chunks.")
    else:
        # Load existing index
        vectorstore = load_vectorstore()
        
    chain = get_qa_chain(vectorstore)
    return chain

# Initialize the RAG chain
try:
    qa_chain = initialize_qa_chain()
except Exception as e:
    st.error(f"An error occurred during initialization: {str(e)}")
    st.stop()

# Input Section
st.markdown('<div class="input-label">💬 What would you like to know?</div>', unsafe_allow_html=True)
query = st.text_input("", placeholder="e.g. What was Swiggy's revenue in FY2024?", label_visibility="collapsed")

if st.button("🔍 Ask Swiggy AI") or query:
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Searching for the answer..."):
            try:
                response = ask(qa_chain, query)
                answer = response.get("answer", "")
                sources = response.get("sources", [])
                
                # Check for "not available" in the response to style appropriately
                is_not_available = "not available" in answer.lower()
                
                # Answer Section
                if is_not_available:
                    st.markdown('<div class="pill-red">❌ Not in Report</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-card answer-card-error">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="pill-green">✅ Answer</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
                
                # Source cards in Expander
                if sources:
                    with st.expander("Supporting Context"):
                        for i, doc in enumerate(sources):
                            page = doc.metadata.get('page', 'Unknown')
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">📄 Page {page}</div>
                                <div class="source-text">{doc.page_content}</div>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to generate an answer: {str(e)}")
