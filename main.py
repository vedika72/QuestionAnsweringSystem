import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- CONFIGURATION ---
DATA_PATH = "./papers"
DB_PATH = "./vector_db"
LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "embeddinggemma:latest"

def load_and_process_documents():
    """
    1. Loads PDFs from the data folder.
    2. Splits them into chunks.
    3. Vectorizes and saves to ChromaDB.
    """
    print("--- 1. Loading Documents ---")
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH}. Please put your PDFs there and run again.")
        return None

    # Load all PDFs from the directory
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found in the 'papers' folder.")
        return None
    
    print(f"Loaded {len(documents)} pages from PDF(s).")

    # Split text into chunks (Optimized for RAG)
    # chunk_size=1000: Good balance for context.
    # chunk_overlap=200: Ensures context isn't lost at cut-off points.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print("--- 2. Creating Vector Store (Embeddings) ---")
    # Initialize Ollama Embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Create and persist ChromaDB
    # This sends text to Ollama -> gets vector -> stores in local folder
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Vector store created and saved locally.")
    return vector_store

def get_vector_store():
    """Checks if DB exists, otherwise creates it."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("Loading existing Vector Store...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print("Initializing new Vector Store...")
        return load_and_process_documents()

def setup_rag_chain(vector_store):
    """
    Sets up the Retrieval Augmented Generation Chain.
    """
    print("--- 3. Setting up Retrieval System ---")
    llm = ChatOllama(model=LLM_MODEL)

    # Convert VectorStore to a Retriever
    # search_type="mmr": Maximal Marginal Relevance (balances similarity with diversity)
    # k=5: Retrieve top 5 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    # Define the Prompt Template
    # We explicitly ask the model to use the context and cite sources.
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI research assistant. Answer the user's question strictly based on the provided context. 
    
    <context>
    {context}
    </context>

    Question: {input}

    Instructions:
    1. If the answer is not in the context, say "I cannot answer this based on the provided documents."
    2. Be concise but detailed.
    3. At the end, you MUST mention the source document names and page numbers based on the metadata.
    """)

    # Create the chain: (Retrieve Context) -> (Combine Context + Question) -> (Pass to LLM)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def query_system(chain, question):
    """
    Process a single question through the chain.
    """
    print(f"\nThinking... (Question: {question})")
    
    response = chain.invoke({"input": question})
    
    print("\n--- ANSWER ---")
    print(response["answer"])
    
    print("\n--- SOURCE ATTRIBUTION ---")
    # Extract metadata from the retrieved documents (context)
    source_docs = response["context"]
    unique_sources = set()
    
    for doc in source_docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        # Cleanup path to just filename
        filename = os.path.basename(source)
        unique_sources.add(f"{filename} (Page {int(page) + 1})")
    
    for source in unique_sources:
        print(f"- {source}")

def main():
    # 1. Initialize DB
    vector_store = get_vector_store()
    
    if not vector_store:
        return

    # 2. Setup Chain
    chain = setup_rag_chain(vector_store)

    # 3. Interactive Loop
    print("\n✅ System Ready! Type 'exit' to quit.\n")
    
    # Pre-defined test questions from your prompt
    sample_questions = [
        "What are the main components of a RAG model?",
        "Explain multi-head attention."
    ]
    
    print(f"Tip: Try asking: '{sample_questions[0]}'")

    while True:
        user_input = input("\n📝 Enter your question: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        query_system(chain, user_input)

if __name__ == "__main__":
    main()