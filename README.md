# LangChain RAG Applications

A collection of RAG (Retrieval Augmented Generation) applications using LangChain, featuring document loading, text splitting, vector stores, and intelligent retrieval.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up .env file
OPENAI_API_KEY=your_key
HUGGINGFACE_ACCESS_TOKEN=your_token

# Run RAG application
python RAG_parts/full_RAG_application.py
```

## 📁 Project Structure

```
├── RAG_parts/              # Core RAG components
│   ├── doct_loader.py     # Document loaders (PDF, Web, Text)
│   ├── text_splitter.py   # Text chunking strategies
│   ├── vector_stores.py   # Vector DB (FAISS, Chroma)
│   ├── retrievers.py      # Retrieval mechanisms
│   └── full_RAG_application.py
├── LLM/                    # LLM examples
├── embedding_models/       # Embedding models
└── chains/                 # LangChain patterns
```

## 💻 Usage

```python
# 1. Load document
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('document.pdf')
doc = loader.load()

# 2. Split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

# 3. Create vector store
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# 4. Query
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
results = retriever.invoke("Your question here")
```

## 🛠️ Technologies

- **LangChain**: Framework for LLM applications
- **Vector Stores**: FAISS, Chroma
- **Embeddings**: OpenAI, Hugging Face
- **LLMs**: GPT-4o-mini, GPT-2
- **UI**: Streamlit

## 📦 Key Dependencies

```
langchain
langchain-community
langchain-openai
langchain-huggingface
faiss-cpu
chromadb
streamlit
youtube-transcript-api
```

## 🎯 Use Cases

- Document Q&A
- YouTube transcript analysis
- Knowledge base search
- Conversational AI




