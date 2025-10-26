from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()
model = HuggingFacePipeline.from_model_id(
    model_id = "gpt2",
    task = "text-generation",
    model_kwargs = {"temperature": 0.0, "max_length": 200}
    )
chat_history = [
    SystemMessage(content = 'you are a AI expert')
]
while True:
    input_text = input("You: ")
    chat_history.append(HumanMessage(content = input_text))
    if input_text == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result))
    print("AI: ",result)
print("Done")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
chat_template = ChatPromptTemplate([
            ('system','You are a helpful customer support agent'),
            MessagesPlaceholder(variable_name = "chat_history"),
            ('human','{query}')
 ])
chat_history = []
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
result = chat_template.invoke({'chat_history': chat_history,'query': "what is refund policy of your company ?"})
print(result)

