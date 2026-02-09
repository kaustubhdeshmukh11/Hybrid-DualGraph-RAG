import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Verify API Key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Please set GROQ_API_KEY in the .env file")

# Path to the company policies document
COMPANY_POLICIES_FILE = "company_policies.txt"

# Initialize LLM globally for reuse
def get_llm():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        temperature=0
    )

def ask_llm_without_context(llm, question):
    """Ask the LLM a question directly without any context (No RAG)"""
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question:\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

def get_baseline_rag_pipeline(file_path=COMPANY_POLICIES_FILE):
    """Initialize RAG pipeline with a local text file"""
    print("--- Initializing Baseline RAG ---")

    # 1. Setup LLM
    llm = get_llm()

    # 2. Load Data from local text file
    print(f"Loading document from {file_path}...")
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # 3. Split Documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks")
    
    # 4. Create Vector Store (Embeddings)
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

    # 5. Define RAG Chain
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever, llm

def demonstrate_why_rag():
    """
    Demonstrate why RAG is needed by:
    1. Asking the LLM a question about fictional company policies (it won't know)
    2. Then asking the SAME question WITH RAG (it will answer correctly)
    """
    print("=" * 70)
    print("       DEMONSTRATION: WHY DO WE NEED RAG?")
    print("=" * 70)
    
    # Questions about fictional Nexora Technologies - LLM definitely won't know these!
    demo_questions = [
        "What is the work from home policy at Nexora Technologies?",
        "How many days of annual leave do employees get at Nexora Technologies?",
        "What is the Star Performer Bonus at Nexora Technologies?",
        "What laptop models are provided to developers at Nexora Technologies?",
    ]
    
    print("\nWe will ask the LLM questions about 'Nexora Technologies' company policies.")
    print("This is a FICTIONAL company - the LLM has never seen this information!")
    print("\nFirst WITHOUT context (no RAG), then WITH context (using RAG).\n")
    
    # Initialize LLM for direct questions
    llm = get_llm()
    
    # Pick the first demo question
    question = demo_questions[0]
    
    print("-" * 70)
    print(f"QUESTION: {question}")
    print("-" * 70)
    
    # Step 1: Ask WITHOUT RAG
    print("\n>>> STEP 1: Asking LLM WITHOUT any context (No RAG)...")
    print("    [The LLM doesn't know about this fictional company]\n")
    
    try:
        response_no_rag = ask_llm_without_context(llm, question)
        print("ANSWER (Without RAG):")
        print("-" * 40)
        print(response_no_rag)
        print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Step 2: Initialize RAG and ask WITH context
    print("\n>>> STEP 2: Now using RAG to retrieve relevant context...")
    print("    [Loading company_policies.txt, creating embeddings]\n")
    
    rag_chain, retriever, _ = get_baseline_rag_pipeline()
    
    print("\n>>> STEP 3: Asking the SAME question WITH RAG context...\n")
    
    try:
        response_with_rag = rag_chain.invoke(question)
        print("ANSWER (With RAG):")
        print("-" * 40)
        print(response_with_rag)
        print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("                         SUMMARY")
    print("=" * 70)
    print("""
WITHOUT RAG:
- The LLM has NO knowledge of 'Nexora Technologies' (fictional company)
- It either admits it doesn't know, or might hallucinate generic policies
- Cannot access private/internal documents

WITH RAG (Retrieval-Augmented Generation):
- The company_policies.txt file is loaded and indexed
- Relevant sections are retrieved based on the question
- LLM answers using ACTUAL policy content (e.g., "3 days WFH per week")
- Works with ANY document without retraining the model!
""")
    print("=" * 70)
    
    # Show a few more example questions
    print("\n>>> BONUS: Try these questions in interactive mode:")
    for i, q in enumerate(demo_questions[1:], 1):
        print(f"   {i}. {q}")

def run_interactive_loop():
    """Run an interactive Q&A loop after the demo"""
    rag_chain, _, _ = get_baseline_rag_pipeline()
    
    print("\n--- RAG System Ready ---")
    print("Ask questions about Nexora Technologies policies.")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        print(f"\nGenerating answer...")
        try:
            response = rag_chain.invoke(query)
            print("\nAnswer:")
            print(response)
            print()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # First, demonstrate why RAG is needed
    demonstrate_why_rag()
    
    # Optionally, continue with interactive mode
    print("\n" + "=" * 70)
    user_choice = input("Would you like to continue with interactive Q&A? (yes/no): ")
    if user_choice.lower() in ["yes", "y"]:
        run_interactive_loop()
    else:
        print("\nThank you for watching the demonstration!")
