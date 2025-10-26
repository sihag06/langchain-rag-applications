from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

loader = PyPDFLoader('attention_all_U_need.pdf')
doc = loader.load()
splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    separator = ''
)
result = splitter.split_documents(doc)

print(len(result))
print(result[0].page_content)
print(len(doc))

loader = PyPDFLoader( 'attention_all_U_need.pdf' )
doc = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

chunks = splitter.split_documents(doc)
print(len(chunks))
print(chunks[0].page_content)

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap = 10
)
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
