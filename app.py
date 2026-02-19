import os
import shutil
import traceback
import re
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

# Print API key status
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"✅ API key loaded: {api_key[:5]}...")
else:
    print("❌ GEMINI_API_KEY not found in .env file!")

app = FastAPI(title="Retail RAG Chatbot - Balanced Retail Assistant")

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create necessary folders
os.makedirs("docs", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

# ---------- BALANCED RETAIL CLASSIFICATION ----------

# Retail keywords - broad enough to catch retail, not too strict
RETAIL_KEYWORDS = {
    "core": [
        "retail", "retailer", "store", "shop", "customer",
        "inventory", "stock", "product", "merchandise",
        "point of sale", "pos", "checkout", "cashier"
    ],
    "operations": [
        "policy", "policies", "procedure", "guideline",
        "return", "refund", "exchange", "warranty",
        "price", "pricing", "discount", "sale", "promotion",
        "employee", "staff", "manager", "schedule"
    ],
    "inventory": [
        "inventory", "stock", "warehouse", "supplier",
        "vendor", "shipment", "delivery", "logistics",
        "reorder", "sku", "barcode", "quantity"
    ]
}

# Non-retail indicators - catch obvious non-retail docs
NON_RETAIL_INDICATORS = [
    # Medical
    "patient", "doctor", "nurse", "hospital", "clinic", "diagnosis",
    "symptom", "treatment", "prescription", "medication", "disease",
    # Resume/CV
    "resume", "curriculum vitae", "cv", "job application", 
    "experience", "education", "university", "college", "degree",
    # Personal
    "my name", "i am", "my address", "phone number", "email address",
    "date of birth", "social security", "passport", "driver license",
    # Recipe
    "recipe", "ingredients", "cook", "bake", "oven", "stove",
    "teaspoon", "tablespoon", "cup", "gram", "kilogram"
]

# Simple conversation memory for follow-up questions
class ConversationMemory:
    def __init__(self):
        self.last_question = None
        self.last_answer = None
        self.topic = None
        self.is_retail_convo = False
    
    def update(self, question, answer, is_retail=True):
        self.last_question = question
        self.last_answer = answer
        self.is_retail_convo = is_retail
        # Extract simple topic (first few words)
        words = question.split()[:4]
        self.topic = " ".join(words)
    
    def is_follow_up(self, question):
        if not self.last_question or not self.is_retail_convo:
            return False
        
        question_lower = question.lower()
        
        # Check for follow-up indicators
        follow_up_patterns = [
            r'\bdefine\s+them\b', r'\bexplain\s+each\b', r'\btell\s+more\b',
            r'\belaborate\b', r'\bcontinue\b', r'\band\b', r'\balso\b',
            r'\bthose\b', r'\bthese\b', r'\bthem\b', r'\bit\b',
            r'\bfirst\b', r'\bsecond\b', r'\bthird\b', r'\blast\b',
            r'\beach\b', r'\bboth\b', r'\ball\b'
        ]
        
        for pattern in follow_up_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Short questions are likely follow-ups
        if len(question.split()) <= 5:
            return True
        
        return False

# Initialize conversation memory
chat_memory = ConversationMemory()

def classify_retail_document(text_content, filename=""):
    """
    Simple but effective retail document classification
    """
    text_lower = text_content.lower()
    filename_lower = filename.lower()
    
    # Count retail vs non-retail indicators
    retail_score = 0
    non_retail_score = 0
    matched_terms = []
    
    # Check filename first (quick check)
    for category, terms in RETAIL_KEYWORDS.items():
        for term in terms:
            if term in filename_lower:
                retail_score += 5
                matched_terms.append(f"filename:{term}")
    
    # Check content
    words = text_content.split()
    total_words = len(words)
    
    for category, terms in RETAIL_KEYWORDS.items():
        for term in terms:
            count = text_lower.count(term)
            if count > 0:
                retail_score += count * 2
                if len(matched_terms) < 10:
                    matched_terms.append(f"{term}({count})")
    
    # Check non-retail indicators
    for term in NON_RETAIL_INDICATORS:
        count = text_lower.count(term)
        if count > 0:
            non_retail_score += count * 3
    
    # Calculate confidence
    total_indicators = retail_score + non_retail_score
    if total_indicators == 0:
        confidence = 0
    else:
        confidence = int((retail_score / (total_indicators + 1)) * 100)
    
    # Decision logic - balanced thresholds
    is_retail = False
    reasons = []
    
    if non_retail_score > retail_score * 2:
        is_retail = False
        reasons.append(f"Strong non-retail signals detected")
    elif retail_score > 10:
        is_retail = True
        reasons.append(f"Found {len(matched_terms)} retail terms")
    elif confidence > 40:
        is_retail = True
        reasons.append(f"Retail confidence: {confidence}%")
    else:
        reasons.append(f"Insufficient retail content")
    
    return is_retail, confidence, reasons, {
        "retail_score": retail_score,
        "non_retail_score": non_retail_score,
        "matched_terms": matched_terms[:5],
        "total_words": total_words
    }

def is_valid_retail_query(query):
    """
    Smart query validation that understands follow-ups and context
    """
    query_lower = query.lower()
    
    # STEP 1: Check if it's a follow-up to previous retail conversation
    if chat_memory.is_follow_up(query):
        print(f"   📌 Detected as follow-up question")
        return True
    
    # STEP 2: Check for retail keywords
    all_retail_terms = []
    for terms in RETAIL_KEYWORDS.values():
        all_retail_terms.extend(terms)
    
    # Add common question patterns
    question_terms = [
        "what", "which", "how", "why", "when", "where",
        "define", "explain", "describe", "list", "tell",
        "can you", "could you", "please", "help"
    ]
    
    for term in all_retail_terms + question_terms:
        if term in query_lower:
            return True
    
    # STEP 3: Very short queries are allowed (likely follow-ups)
    if len(query.split()) <= 3:
        return True
    
    # STEP 4: Check if query references previous topic
    if chat_memory.topic:
        topic_words = chat_memory.topic.lower().split()
        for word in topic_words:
            if len(word) > 3 and word in query_lower:
                return True
    
    return False

# ---------- PDF Processing Functions ----------

def load_pdf(file_path):
    print(f"📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"   → {len(pages)} pages loaded")
    return pages

def extract_full_text(pages):
    """Extract all text from pages for classification"""
    return " ".join([page.page_content for page in pages])

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
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

# ---------- API Endpoints ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"\n📤 Upload endpoint called: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = f"docs/{file.filename}"
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"   ✅ File saved to {file_path}")
        
        # Load PDF and extract text
        pages = load_pdf(file_path)
        full_text = extract_full_text(pages)
        
        # Classify document
        print("🔍 Classifying document...")
        is_retail, confidence, reasons, details = classify_retail_document(full_text, file.filename)
        
        print(f"\n📊 Classification: {'RETAIL' if is_retail else 'NON-RETAIL'}")
        print(f"   Confidence: {confidence}%")
        print(f"   Reasons: {', '.join(reasons)}")
        
        # Reject non-retail documents
        if not is_retail and confidence < 30:
            print(f"\n❌ Document rejected: Not retail-related")
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise HTTPException(
                status_code=400,
                detail=(
                    f"❌ '{file.filename}' does not appear to be a retail document.\n\n"
                    f"This system only accepts retail-related documents like:\n"
                    f"• Store policies & procedures\n"
                    f"• Inventory reports\n"
                    f"• Pricing guides\n"
                    f"• Sales data\n"
                    f"• Retail operations manuals\n\n"
                    f"Please upload a retail document."
                )
            )
        
        # Process retail document
        print(f"\n✅ Document accepted - Processing...")
        chunks = chunk_documents(pages)
        embeddings = get_embedding_model()
        vectorstore = create_vector_store(chunks, embeddings)
        save_vector_store(vectorstore)
        
        print("   ✅ Upload complete\n")
        
        return {
            "message": f"✅ '{file.filename}' uploaded successfully",
            "retail_classification": {
                "is_retail": True,
                "confidence": confidence,
                "pages": len(pages),
                "chunks": len(chunks)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ Error: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat")
async def chat(query: str = Form(...)):
    print(f"\n💬 Chat Query: '{query}'")
    
    # Validate query (with follow-up support)
    if not is_valid_retail_query(query):
        print("   ❌ Query rejected - not retail-related")
        return {
            "answer": (
                "I'm a retail document assistant. "
                "Please ask questions about retail topics like:\n"
                "• Store policies and procedures\n"
                "• Inventory and stock management\n"
                "• Pricing and promotions\n"
                "• Customer service guidelines\n"
                "• Retail operations"
            ),
            "sources": []
        }
    
    try:
        # Load vector store
        embeddings = get_embedding_model()
        vectorstore = load_vector_store(embeddings)
        
        if vectorstore is None:
            return {
                "answer": "Please upload a retail document first.",
                "sources": []
            }
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Get top 4 chunks
        )
        
        # Retrieve relevant chunks
        docs = retriever.invoke(query)
        
        if not docs:
            return {
                "answer": "I couldn't find information about that in your document. Try rephrasing or asking about something else.",
                "sources": []
            }
        
        # Format context with page numbers
        def format_docs(docs):
            return "\n\n---\n\n".join([
                f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}" 
                for doc in docs
            ])
        
        # Create prompt - balanced, not too strict
        template = """You are a helpful retail document assistant. Answer questions based ONLY on the provided document context.

Context from retail document:
{context}

Question: {question}

Guidelines:
- Use ONLY the context above to answer
- If the context doesn't contain the answer, say "I don't see that information in the document"
- Be concise but informative
- Use bullet points for lists when appropriate

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        chain = (
            RunnableParallel(
                context=retriever | format_docs,
                question=RunnablePassthrough()
            )
            | prompt
            | get_llm()
            | StrOutputParser()
        )
        
        # Get answer
        answer = chain.invoke(query)
        
        # Extract sources
        sources = list(set([
            doc.metadata.get("page", 0) for doc in docs 
            if "page" in doc.metadata
        ]))
        
        # Update conversation memory
        chat_memory.update(query, answer, is_retail=True)
        
        print(f"   ✅ Responded with {len(answer)} chars")
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "memory": chat_memory.topic}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)