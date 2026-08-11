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
    """Loads and embeds documents using native Python, saving them to disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # 1. Load instantly if the database already exists
    if os.path.exists(persist_dir):
        print("Loading existing vector database from disk...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectorstore.as_retriever()
        
    print("Building and embedding new vector database...")
    
    # 2. Native Python file loading (This completely fixes the DeprecationWarning)
    with open(file_path, "r", encoding="utf-8") as file:
        text_content = file.read()
    
    docs = [Document(page_content=text_content, metadata={"source": file_path})]

    # 3. Split and persist
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    retriever = initialize_vector_store("data/knowledge_base.txt")
    
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
        if input_dict.get("chat_history"):
            return standalone_q_chain.invoke(input_dict)
        return input_dict["input"]

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