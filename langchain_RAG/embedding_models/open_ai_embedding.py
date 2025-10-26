from langchain_openai import OpenAIEmbeddings
from dotenv imoprt load_dotenv

load_dot_env()
#this is the openai embedding model which is used to embed a query into a vector space
embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimentsions = 32)
embeddings.embed_query("hello, how are you?")
print(str(embeddings))

# this is the openai embedding model which is used to embed a document into a vector space
embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimentsions = 32)
text = ["Indian tech startups are failed to invate any new innovation",
"what about USE on other hand leading the world in innovation",
"china is also doing good work in field of AI and machine learning "
]
response = embedding.embed_documents(text)
print(str(response))


