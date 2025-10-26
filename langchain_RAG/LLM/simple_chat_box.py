from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
import os
from dotenv import load_dotenv

load_dotenv()
model = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    model_kwargs={"temperature": 0.0, "max_length": 200}
)
messages  = [
    SystemMessage(content = "you an expert of AI field"),
    HumanMessage(content = "what is langgraph?")
]

result = model.invoke(messages)

# Add the AI response to messages
messages.append(AIMessage(content=result))
print(messages)