import os
from dotenv import load_dotenv

# Modern Core Imports (Notice we removed langchain_community entirely)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Modern LCEL Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def initialize_vector_store(file_path: str, persist_dir: str = "./chroma_db"):
    """Loads the CMU Book Summary Dataset (booksummaries.txt)."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if os.path.exists(persist_dir):
        print("Loading existing library database from disk...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 2})
        
    print("Cataloging CMU dataset into vector database (this may take a few minutes)...")
    
    docs = []
    # Native Python CSV loading configured for Tab-Separated Values (TSV)
    with open(file_path, mode="r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        
        for row in tsv_reader:
            # The CMU dataset has 7 columns; we skip any malformed rows
            if len(row) < 7:
                continue
                
            content = row[6] # The Plot Summary
            
            # Map the metadata using the strict column indices
            metadata = {
                "title": row[2] if row[2] else "Unknown Title",
                "author": row[3] if row[3] else "Unknown Author",
                "year": row[4] if row[4] else "Unknown Year"
            }
            
            # Ensure we don't embed empty strings
            if content.strip():
                docs.append(Document(page_content=content, metadata=metadata))
                
    # --- IMPORTANT PORTFOLIO TIP ---
    # The CMU dataset has over 16,500 books. Embedding all of them via the API 
    # will take a long time and might hit free-tier rate limits. 
    # Uncomment the line below to test the pipeline with just the first 500 books first:
    # docs = docs[:500] 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    # Return the top 2 closest book matches
    return vectorstore.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    retriever = initialize_vector_store("data/booksummaries.txt")
    
    # Swapped to 3.5-flash to regain control and prevent the 400 Crash
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)

    condense_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "formulate a standalone question which can be understood "
                   "without the chat history. Do NOT answer the question, "
                   "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    standalone_q_chain = condense_q_prompt | llm | StrOutputParser()

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly capable assistant. Use the following pieces of "
                   "retrieved context to answer the question accurately.\n\nContext: {context}"),
        ("human", "{standalone_question}"),
    ])

    def route_question(input_dict):
        # 1. If there is no chat history, just use the raw input
        if not input_dict.get("chat_history"):
            return input_dict["input"]
            
        # 2. If there IS chat history, ask the AI to rewrite the question
        rewritten_q = standalone_q_chain.invoke(input_dict)
        rewritten_q = rewritten_q.strip() # Remove accidental blank spaces
        
        # Print it to the terminal so we can watch the AI's "thought process"
        print(f"\n[DEBUG] AI rewrote your question for the database search as: '{rewritten_q}'")
        
        # 3. THE SAFETY CATCH: If the AI returns an empty string, fallback to the original input
        if not rewritten_q:
            print("[DEBUG] The rewritten question was empty. Falling back to original input.")
            return input_dict["input"]
            
        return rewritten_q

    rag_chain = (
        RunnablePassthrough.assign(
            standalone_question=route_question
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    chat_history = []
    print("\n--- Modern LCEL Gemini RAG Bot Initialized ---")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        print(f"\nGemini: {response}\n")
        
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()