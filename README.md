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

https://github.com/harSHITags/Retail-RAG-Chatbot/blob/main/RetailBot%20-HLD%20Document.pdf

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
   git clone https://github.com/harSHITags/Retail-RAG-Chatbot
   cd Retail-Chatbot
   ```

### 2. Setup Environment Variables
Create a `.env` file in the root directory and add:
```
GEMINI_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
uvicorn app:app --reload
```

## 🤖 Autonomous Research Mode
This application now features an **Autonomous Research Agent** powered by CrewAI and DuckDuckGo search.
When a user asks a complex retail question, the research mode will:
1. Search the live internet for up-to-date facts.
2. Delegate tasks to a Senior Retail Analyst agent.
3. Synthesize findings into a final report using a Strategy Consultant agent.
4. Save the detailed report directly to the `knowledge_repository/research_findings/` folder for persistent storage.

### Example Queries
- "What are the latest self-checkout theft prevention strategies in 2024?"
- "Compare the inventory management practices of Walmart vs Target."
- "What are the top e-commerce fulfillment trends for retail this year?"