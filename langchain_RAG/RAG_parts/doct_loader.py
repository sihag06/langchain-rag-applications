from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from dotenv import load_dotenv

load_dotenv()
loader = TextLoader('cricket_poem.txt')
doc = loader.load()
print(len(doc))
print(doc[0].page_content)

loader = PyPDFLoader('attention_all_U_need.pdf')
doc = loader.load()
print(doc[0].page_content)
url = 'https://en.wikipedia.org/wiki/Ek_Deewane_Ki_Deewaniyat'
loader = WebBaseLoader(url)
doc = loader.load()
print(len(doc))
# print(doc[0].page_content)
print(doc[0].metadata)