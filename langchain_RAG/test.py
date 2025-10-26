import langchain
print(langchain.__version__)
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
llm = OpenAI(model = 'gpt-3.5-turbo',max_completion_tokens= 50)
result = llm.invoke("what's the wheather of IIT Roorkee?")
print(result)

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
model = ChatOpenAI(model = 'gpt-3.5-turbo')
resutl = model.invoke(HumanMessage(content = "give me a code to check whether a number is prime or not"), temperature = 0.3)
print(result)


from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = llm.invoke("Hello, how are you?")
print(result)

from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model = 'claude-3-5-sonnet-latest')
result = llm.invoke("give me a code to check whether a number is prime or not")
print(result)






