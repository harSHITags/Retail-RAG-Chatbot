# Retail RAG Chatbot

A powerful **Retrieval-Augmented Generation (RAG)** system for retail document intelligence. Upload retail PDFs and ask questions – answers are grounded strictly in your documents with zero hallucination.

![Demo](https://via.placeholder.com/800x400?text=Retail+RAG+Chatbot+Demo)

##  Features

-  **PDF Upload** – Upload retail documents (policies, reports, manuals)
-  **Intelligent Retrieval** – Finds the most relevant sections
-  **Gemini AI Integration** – Powered by Google's latest models
-  **No Hallucination** – Strict prompting ensures answers come only from your docs
-  **Retail Classification** – Automatically rejects non-retail documents
-  **Source Citations** – Shows which pages answers come from
-  **Beautiful UI** – Modern, responsive design
-  **Privacy First** – All processing happens locally (except LLM calls)

##  Architecture

![Architecture Diagram](https://via.placeholder.com/800x400?text=Architecture+Diagram)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python) |
| Frontend | HTML, CSS, JavaScript |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | FAISS |
| LLM | Google Gemini 3 Flash |
| PDF Processing | PyPDFLoader |
| Text Chunking | LangChain |

## Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/harSHITags/retail-rag-chatbot.git
   cd retail-rag-chatbot