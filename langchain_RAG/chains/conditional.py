from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

#from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model ='gpt-4o-mini')
parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description = 'Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = Feedback)
prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into postive or negative {feedback}{format_instructions}',
    input_variables = ['feedback'],
    partial_variables = {'format_instructions':parser2.get_format_instructions()}

)
classification_chain = prompt1 | model |parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback{feedback}',
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback{feedback}',
    input_variables = ['feedback']
)
response_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | model |parser1),
    (lambda x : x.sentiment == 'negative', prompt3 | model |parser1),
    RunnableLambda(lambda x : 'failed to find sentiment')

)
chain = classification_chain | response_chain
result = chain.invoke({'feedback' : 'I love the product and the service is great!'})
print(result)