# 📚 BiblioBot: AI Librarian (Semantic Book Search Engine)

BiblioBot is a full-stack, conversational Retrieval-Augmented Generation (RAG) application that acts as an expert librarian. Users can query the bot with vague plot descriptions, themes, or character details, and the AI uses semantic vector search to identify the exact book from a local database and provide a natural language explanation of why it matches.

## 🚀 Features
* **Semantic Search:** Replaces traditional keyword matching with vector-based semantic similarity to find books based on abstract plot points.
* **Conversational Memory:** Maintains chat history context, dynamically reformulating follow-up questions for accurate database querying.
* **Modern LCEL Architecture:** Built using LangChain Expression Language (LCEL) for a clean, modular, and declarative pipeline.
* **Local Vector Database:** Utilizes ChromaDB for high-speed, on-disk document retrieval without relying on expensive cloud database solutions.
* **Interactive UI:** Features a lightweight, responsive web interface built with Streamlit.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** LangChain (LCEL)
* **LLM & Embeddings:** Google Gemini 3.5 Flash & Gemini Embedding-001
* **Vector Database:** ChromaDB 
* **Frontend:** Streamlit
* **Dataset:** CMU Book Summary Dataset (Tab-Separated Values)

## 🏗️ Architecture Flow
1. **Ingestion & Chunking:** Reads the CMU Book Summary dataset, separating plot summaries as embedded content and isolating Book Titles and Authors as searchable metadata. Text is optimized using `RecursiveCharacterTextSplitter`.
2. **Vectorization:** Converts chunked summaries into mathematical vectors using Google's generative AI embedding models and persists them locally via Chroma.
3. **Query Reformulation:** Intercepts user input and previous chat history, rewriting the query to be completely standalone.
4. **Retrieval & Generation:** Performs a similarity search in ChromaDB, injects the top matching documents (along with metadata) into a custom prompt template, and streams the final synthesis through the Gemini LLM.

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/Hammad-Ghaznavi/Langchain-RAG.git](https://github.com/Hammad-Ghaznavi/Langchain-RAG.git)
cd Langchain-RAG
```

**2. Set up the virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and add your Google AI Studio API key:
```env
GOOGLE_API_KEY=your_api_key_here
```

**5. Add the Dataset**
* Download the **CMU Book Summary Dataset** (`booksummaries.txt`).
* Place the file inside a `data/` folder in the root directory (`data/booksummaries.txt`).

**6. Launch the Application**
```bash
streamlit run app.py
```
*Note: On the very first run, the application will take a moment to embed the dataset and build the local `chroma_db` folder. Subsequent launches will load the database instantly from disk.*

---
**Author:** [Hammad Ghaznavi](https://github.com/Hammad-Ghaznavi)