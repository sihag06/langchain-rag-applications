from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
text = ["Indian tech startups are failed to innovate any new innovation",
"what about USE on other hand leading the world in innovation",
"china is also doing good work in field of AI and machine learning "
]
response = embedding.embed_documents(text)
print(str(response))

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
text = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query = "who is MS Dhoni?"
doc_response = embedding.embed_documents(text)
query_response = embedding.embed_query(query)

# ✅ Convert query to 2D
query_response = np.array(query_response).reshape(1, -1)

# ✅ Convert docs to numpy array
doc_response = np.array(doc_response)
scores = cosine_similarity(query_response, doc_response)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(text[index])
print("similarity score is:", score)
