import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# ONLY import the global functions from main
from main import initialize_vector_store, format_docs

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Page Configuration
st.set_page_config(page_title="AI Librarian", page_icon="📚")
st.title("📚 BiblioBot: Your AI Librarian")
st.markdown("Describe a book you vaguely remember, and I'll find it for you!")

# 2. Wrap heavy resource loading in a Streamlit spinner
@st.cache_resource
def load_retriever():
    return initialize_vector_store("data/booksummaries.txt")

with st.spinner("Booting up vector database and AI librarian models..."):
    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)

    # 3. Define the Chains and Prompts specifically for the Web UI
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
        ("system", "You are an expert, friendly librarian at a help desk. Your goal is to help users find books based on vague descriptions, plot points, or themes.\n\n"
                   "Use the retrieved book summaries below to identify the book they are looking for. "
                   "When you suggest a book, you MUST explicitly state the Title and Author (which are provided in the context metadata), "
                   "and give a brief explanation of why it matches their query.\n\n"
                   "If the provided context does not contain a matching book, kindly say that it is not in our current catalog.\n\n"
                   "Retrieved Catalog Context:\n{context}"),
        ("human", "{standalone_question}"),
    ])

# 4. Rebuild the LCEL Routing
def route_question(input_dict):
    if input_dict.get("chat_history"):
        rewritten_q = standalone_q_chain.invoke(input_dict).strip()
        if not rewritten_q:
            return input_dict["input"]
        return rewritten_q
    return input_dict["input"]

rag_chain = (
    RunnablePassthrough.assign(standalone_question=route_question)
    | RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["standalone_question"])))
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 5. Manage Chat History in Streamlit Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# Display previous messages
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Handle User Input
if user_input := st.chat_input("E.g., 'What's that book where kids go to battle school in space?'"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.display_history.append({"role": "user", "content": user_input})

    with st.spinner("Searching the catalog..."):
        # --- NEW DEBUG CODE ---
        # This will print the exact books it found to your VS Code terminal
        raw_docs = retriever.invoke(user_input)
        print("\n--- [DEBUG] DATABASE RESULTS ---")
        for i, doc in enumerate(raw_docs):
            print(f"Match {i+1}: {doc.metadata.get('title', 'Unknown')} by {doc.metadata.get('author', 'Unknown')}")
        print("--------------------------------\n")
        # ----------------------
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.display_history.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))