from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt1 = PromptTemplate(
    template = 'write down a short repoet on topic {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'write a 3 facts from the given text {text}',
    input_variables = ['text']
)
model = ChatOpenAI(model = 'gpt-4o-mini')
parser = StrOutputParser()
chain = prompt1 | model | parser |prompt2 |model |parser
print(chain.invoke({'topic' : 'role of machine learning in agricuture'}))
