from langchain_openai import ChatopenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda, RunnableParallel
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model = 'gpt-4o-mini')
parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = 'generate a detailed report on the topic {topic}',
    input_variables = {'topic'}
)
prompt2 = PromptTemplate(
    template = 'generate a summary of the following text {text}',
    input_variables = {'text'}
)

report_chain = prompt1 | model | parser

branch = RunnableBranch(
         lambda x : len(x.split())>300, prompt2 | model | parser
         Runnablepassthrough() 
)
final_chain = report_chain | branch
result = final_chain.invoke({'topic': 'molecular bilogy'})
print(result)
