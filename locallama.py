import pickle
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries. If you don't know the answer, say don't know "),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title('Langchain API Demo With LLAMA2 ')

# Load chat history from file (if it exists)
try:
    with open('chat_history.pkl', 'rb') as f:
        chat_history = pickle.load(f)
except FileNotFoundError:
    chat_history = []

input_text=st.text_input("Search the topic u want")

# ollama LLAma2 LLm
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    chat_history.append((input_text, chain.invoke({"question":input_text})))
    st.write("You: ", input_text)
    st.write("Assistant: ", chat_history[-1][-1])
    for old_query, old_response in chat_history[:-1]:
        st.write("You: ", old_query)
        st.write("Assistant: ", old_response)

# Save chat history to file
with open('chat_history.pkl', 'wb') as f:
    pickle.dump(chat_history, f)