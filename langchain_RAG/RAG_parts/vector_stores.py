# from langchain.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

import os
from dotenv import load_dotenv
load_dotenv()

# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
docs = [ doc1, doc2, doc3, doc4, doc5 ]
embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    embedding_function = embedding,
    persist_directory = "chroma_db",
    collection_name = "ipl_players"
)
#add documents
print(vector_store.add_documents(docs))

#view documents and embeddings
vector_store.get(include = ['documents','embeddings'])

search for document
result1 = vector_store.similarity_search("who is Virat Kohli?",
k =1)
print(result1)
result2 = vector_store.similarity_search_with_score('Who among these are a bowler?',k = 2)
print(result2)
update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(documents_id = '91e895e3-056a-4456-b239-3fd739a3b88e' , document = updated_doc1)
# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])
# delete document
vector_store.delete(ids=['c38094ae-7341-4d6f-9be0-d6a8e5a18268'])
# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])
