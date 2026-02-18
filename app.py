import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# LCEL imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

# Print API key status (first 5 chars only)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"✅ API key loaded: {api_key[:5]}...")
else:
    print("❌ GEMINI_API_KEY not found in .env file!")

app = FastAPI(title="Retail RAG Chatbot")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create necessary folders
os.makedirs("docs", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

# ---------- Helper Functions ----------

def load_pdf(file_path):
    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"   → {len(pages)} pages loaded")
    return pages

def chunk_documents(documents):
    print(f"✂️ Chunking {len(documents)} pages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks created")
    return chunks

def get_embedding_model():
    print("🔧 Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vector_store(chunks, embeddings):
    print("💾 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("   → Vector store created")
    return vectorstore

def save_vector_store(vectorstore, path="faiss_index"):
    print(f"💿 Saving vector store to {path}...")
    vectorstore.save_local(path)
    print("   → Saved")

def load_vector_store(embeddings, path="faiss_index"):
    print(f"📂 Loading vector store from {path}...")
    if not os.path.exists(path):
        print(f"   ❌ {path} does not exist!")
        return None
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print("   → Loaded")
    return vectorstore

def get_llm():
    print("🤖 Initializing Gemini LLM...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",                     # <-- changed
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

# ---------- API Endpoints ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"\n📤 Upload endpoint called: {file.filename}")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = f"docs/{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"   ✅ File saved to {file_path}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    try:
        pages = load_pdf(file_path)
        chunks = chunk_documents(pages)
        embeddings = get_embedding_model()
        vectorstore = create_vector_store(chunks, embeddings)
        save_vector_store(vectorstore)
        print("   ✅ Upload and processing complete\n")
        return {"message": f"File '{file.filename}' uploaded and processed successfully"}
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        traceback.print_exc()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/chat")
async def chat(query: str = Form(...)):
    print(f"\n💬 Chat endpoint called. Query: '{query}'")
    try:
        # Step 1: Load embeddings and vector store
        print("Step 1: Loading embedding model...")
        embeddings = get_embedding_model()
        
        print("Step 2: Loading vector store...")
        vectorstore = load_vector_store(embeddings)
        if vectorstore is None:
            print("   ❌ No vector store found. Please upload a PDF first.")
            raise HTTPException(status_code=400, detail="No documents uploaded yet. Please upload a PDF first.")
        
        print("Step 3: Creating retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Step 4: Retrieve relevant docs
        print("Step 4: Retrieving relevant chunks...")
        docs = retriever.invoke(query)
        print(f"   → Retrieved {len(docs)} chunks")
        
        # Step 5: Build prompt
        print("Step 5: Building prompt...")
        template = """You are a retail document assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say "I cannot find this information in the uploaded documents."

Context:
{context}

Question: {question}
Helpful Answer:"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # Step 6: Initialize LLM
        print("Step 6: Initializing LLM...")
        llm = get_llm()
        
        # Step 7: Format docs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Step 8: Create chain
        print("Step 7: Creating RAG chain...")
        chain = (
            RunnableParallel(
                context=retriever | format_docs,
                question=RunnablePassthrough()
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Step 9: Invoke chain
        print("Step 8: Invoking chain...")
        answer = chain.invoke(query)
        print(f"   → Answer received: {answer[:50]}...")
        
        # Step 10: Extract sources
        sources = list(set(
            doc.metadata.get("page", 0) for doc in docs if "page" in doc.metadata
        ))
        print(f"   → Sources: {sources}")
        
        print("✅ Chat completed successfully\n")
        return {"answer": answer, "sources": sources}
        
    except HTTPException:
        raise  # re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"\n❌ Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)