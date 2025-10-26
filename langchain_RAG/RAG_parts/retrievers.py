from langchain_community.retrievers import WikipediaRetriever
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
#source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
#create Chrooma Vector store 
vector_store = Chroma.from_documents(
    documents = documents,
    embedding = embedding,
    collection_name = "langchain_docs"
)
retriver = vector_store.as_retriever(search_kwargs = {"k":2})
query = "What is Chroma used for?"
result = retriver.invoke(query)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

#MMR
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
embedding_model = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(
    documents = docs,
    embedding = embedding_model
)
# Enable MMR in the retriever
retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {"k":2, "lambda_mult":0.5}
)
query = "What is langchain?"
results = retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)