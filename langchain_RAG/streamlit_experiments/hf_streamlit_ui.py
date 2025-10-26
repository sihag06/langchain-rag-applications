from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import load_prompt

load_dotenv()

# Set up model
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
model = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    model_kwargs={"temperature": 0.0, "max_length": 200}
)

st.header("Hugging Face Chatbot on research paper explanation")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 
length_input = st.selectbox("Select explanation length", ["short", "medium", "long"])

template = load_prompt('template.json')

if st.button("Summarize Paper"):
    with st.spinner("Generating explanation..."):
        chain = template | model
        result = chain.invoke({
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        })
        st.write(result)