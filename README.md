# 🍔 Swiggy Annual Report — RAG Q&A System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-blueviolet)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-orange)

## 1. Overview
This project is an AI-powered Question Answering system built entirely on Swiggy's official Annual Report using Retrieval-Augmented Generation (RAG). The system is strictly engineered to answer queries based *only* on the provided document and is designed to never hallucinate outside information. It was built as a demonstration of production-ready RAG pipeline design, featuring optimized chunking, local embeddings, and a custom branded UI.

## 2. Source Document
* **Document**: Swiggy Annual Report FY2024
* **Source Link**: [https://investors.swiggy.com/annual-reports](https://investors.swiggy.com/annual-reports)
* **Format**: PDF
* *Note: The PDF is not included in this repository due to file size constraints and must be downloaded manually.*

## 3. The Problem & Approach

**The Core Challenge:** Large PDF documents cannot be directly fed into Large Language Models (LLMs) due to context window limits and hallucination risks.

**The Memory & Context Window Problem**
LLMs have a limited context window (typically ranging from 4k to 128k tokens). A 200+ page annual report is far too large to fit into this window. Even if a model supported a massive context window, feeding the entire document would cause the model to struggle with focusing on relevant parts ("lost in the middle" syndrome) and would be prohibitively expensive to run on every single query.

**The RAG Solution**
Instead of feeding the whole document at once, we pre-process it once and store it as searchable vector embeddings. At query time, only the most relevant 5 chunks (out of thousands) are retrieved and sent to the LLM. This means the LLM only ever sees a small, highly-focused context window of relevant information. This makes the system accurate, exceptionally fast, and cost-efficient.

**The Anti-Hallucination Strategy**
The prompt strictly forbids the LLM from using any knowledge outside the retrieved context. If the answer cannot be found in the retrieved chunks, the system is programmed to respond with a fixed fallback message ("This information is not available in the Swiggy Annual Report.") instead of attempting to guess or hallucinate an answer.

**The Chunking Strategy Decision**
Chunks of 600 characters with a 50-character overlap were chosen deliberately. If chunks are too large, they lose semantic focus, making the vector search less precise. If they are too small, important surrounding context gets cut off. The overlap of 50 characters ensures that sentences and concepts are never cleanly cut at boundaries. Furthermore, duplicate chunks and junk fragments under 50 characters are aggressively filtered out before indexing to reduce database noise and save storage space.

**The Embedding Model Choice**
The `sentence-transformers/paraphrase-MiniLM-L3-v2` model was chosen for embedding generation because it runs entirely locally on the CPU with zero API costs. At just 45MB in size, it is incredibly lightweight while still performing exceptionally well on semantic similarity tasks for English business text.

**The Vector Database Choice**
FAISS (Facebook AI Similarity Search) was chosen over cloud vector databases like Pinecone because it runs entirely locally. It requires no server, incurs no cost, and introduces zero network latency. It is more than fast enough for a single-document use case, and the index is saved directly to disk so the heavy lifting only needs to be processed once.

## 4. Technology Stack

| Component | Tool / Library | Reason |
| :--- | :--- | :--- |
| **PDF Loading** | PyMuPDF (`fitz`) | Fast and highly accurate text and metadata extraction. |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Handles structure-aware splitting with precise overlaps. |
| **Embeddings** | `paraphrase-MiniLM-L3-v2` | Lightweight, free, and runs entirely locally. |
| **Vector Store** | FAISS | Local, lightning-fast in-memory search, no server required. |
| **LLM** | Groq (`llama-3.1-8b-instant` / LLaMA 3.3) | Utilizes the generous free tier for blazing fast inference. |
| **User Interface** | Streamlit | Rapid prototyping with a clean, customizable Python web interface. |
| **Environment** | `python-dotenv` | Secure and simple API key management. |

## 5. Architecture Flow

```text
[ RUNTIME INFERENCE FLOW ]
User Query 
    │
    ▼
Embedding Model (MiniLM) ──► Query Vector
                                │
                                ▼
                        FAISS Vector Search
                                │
                                ▼
                       Retrieves Top 5 Chunks
                                │
                                ▼
                      Strict Prompt Template
                                │
                                ▼
                Groq LLM (LLaMA 3.1 / 3.3 via API)
                                │
                                ▼
                   Final Answer Displayed in UI


[ ONE-TIME INDEXING FLOW ]
PDF Document
    │
    ▼
PyMuPDF (Extract Text & Page Metadata)
    │
    ▼
Clean & Chunk (600 chars, 50 overlap)
    │
    ▼
Deduplicate (Remove <50 chars & exact matches)
    │
    ▼
Embed via MiniLM-L3-v2
    │
    ▼
FAISS Index (Saved to disk: faiss_index/)
```

## 6. Project Structure

```text
swiggy-rag/
├── data/                          # Folder where the PDF report must be placed
├── faiss_index/                   # Auto-generated local vector database
├── src/
│   ├── __init__.py                # Python package marker
│   ├── document_processor.py      # PDF extraction, cleaning, and chunking logic
│   ├── embeddings.py              # HF embeddings and FAISS vectorstore management
│   └── rag_pipeline.py            # Langchain prompt, retriever, and Groq LLM setup
├── app.py                         # Streamlit frontend with custom Swiggy CSS branding
├── requirements.txt               # Dependencies list
├── .env.example                   # Template for environment variables
└── README.md                      # Project documentation (You are here)
```

## 7. Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd swiggy-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a Groq API Key**:
   * Visit [https://console.groq.com](https://console.groq.com) and create a free account to generate an API key.

4. **Configure Environment Variables**:
   * Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   * Open `.env` and assign your key: `GROQ_API_KEY=your_actual_key`

5. **Download the Swiggy Annual Report**:
   * Download the PDF from the official investors page.
   * Place it exactly at `data/swiggy_annual_report.pdf`.

6. **Run the Application**:
   ```bash
   python -m streamlit run app.py
   ```
   *Note: The very first run will process the PDF and build the FAISS index, which takes 1-2 minutes. All subsequent runs will load instantly.*

## 8. Example Questions to Try

1. **Financials**: "What was Swiggy's total revenue from operations in FY2024?"
2. **Financials**: "How much did the company spend on advertising and sales promotion?"
3. **Board Members**: "Who are the independent directors on the board?"
4. **Board Members**: "What degree does Lakshmi Nandan Reddy Obul hold?"
5. **Business Segments**: "What are the primary business segments Swiggy operates in?"
6. **Business Segments**: "How is Instamart performing compared to food delivery?"
7. **Company Strategy**: "What are the key risk factors mentioned in the report?"
8. **Company Strategy**: "What initiatives is Swiggy taking regarding sustainability and ESG?"

## 9. Key Design Decisions

* **k=5 Retrieval Chunks**: Retrieving exactly 5 chunks provides enough surrounding context for the LLM to synthesize a complete answer without overflowing the context window or diluting the prompt with irrelevant noise.
* **Temperature = 0**: The LLM's temperature is set to absolute zero to enforce deterministic, highly analytical responses. We do not want the model being "creative" when answering factual financial questions.
* **PDF Excluded from Repo**: Annual reports are typically very large binary files (often 10MB+). Committing them to Git bloats the repository history unnecessarily.
* **Overlap Reduction (150 to 50)**: During optimization, the chunk overlap was reduced from 150 to 50 characters to save significant disk space and memory footprint in the vector store while still safely preventing hard word-cuts at chunk boundaries.

## 10. Limitations

* **Temporal Knowledge**: The system cannot answer questions about events, earnings, or news that occurred after the report was published.
* **Complex Formatting**: Highly complex tables, charts, or scanned image pages within the PDF may not extract perfectly via PyMuPDF, potentially causing the model to miss tabular data context.
* **Multi-Part Complexity**: Extremely complex, multi-part questions that require synthesizing data from dozens of disconnected pages may only receive partial answers since only the top 5 chunks are retrieved.
* **API Limits**: The free tier of the Groq API has strict rate limits (Tokens per Minute / Requests per Minute) which may occasionally throw a `429 RESOURCE_EXHAUSTED` error if queried too rapidly.

## 11. License

This project is licensed under the MIT License.
