from langchain_openai import ChatOpenAI
from langchain_core.huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema.runnable import Runnablepallel
from langchain_core.prompts import PromptTemplate
from langchain_core.optput_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI(model = 'gpt-40-mini')
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"

)
model2 = ChatHuggingFace(llm =llm)

prompt1 = PromptTemplate(
          template = 'generate a simple notes from given text {text},
          input_variables = ['text']
)
prompt2 = PromptTemplate(
    template = 'generate a 3 question for Quix from given test {text},
    input_variables = ['text']
)
prompt3 = PromptTemplate(
          template = 'merge the notes with the quiz in a single documents {notes}{quiz}',
          input_varibles = ['notes','quiz']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel(
    'notes': prompt1 | model1 |parser,
    'quiz': prompt2 | model2 |parser
)
merge_chain = prompt3 | model1 |parser
final_chain = parallel_chain | merge_chain
text = """MFCC:-
MFCC is a technique to extract features from an audio signal by converting each small audio frame into a vector that captures the most important characteristics of the sound, especially how humans perceive it.
So for each short audio frame (~20-40 ms), you end up with a 13-dimensional feature vector representing that slice of sound.
Cepstral coefficients = the actual feature numbers that describe a sound frame.
In MFCC, we usually extract 12-13 cepstral coefficients per frame."""
result = final_chain.invoke({'text': text}) 
print(resutl)

final_chain.get_graph().print_ascii()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model = 'gpt-4o-mini')
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = 'generate a simpel instagram post on a topic{topic}',
    input_variable = {'topic'}
)
prompt2 = PromptTemplate(
    template = 'generate a tweet on a topic{'topic'}',
    input_variable = {'topic'}
)
chain = RunnableParallel({
    'instagram' : prompt1 | model |parser,
     'tweet' : prompt2 | model | parser
})
print(chain.invoke({'topic': 'AI'}))

