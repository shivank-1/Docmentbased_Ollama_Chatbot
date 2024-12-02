from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

##Langsmith Tracking
os.environ["langchain_api_key"]=os.getenv("langchain_api_key")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot with Ollama"

##Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You're a helpful assistant. Please respond to the user queries"),
        ("user","Questions{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm=Ollama(model=engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({"question":question})
    return answer


##Select a model
engine=st.sidebar.selectbox("Select Your Model",["Llama3","gemma2"])

##Title of the app
st.title('Enhanced Q&A Chatbot With GROQ')

##Sidebar for settings
st.sidebar.title("Settings")


##Adjust reponse parameter 
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max_tokens",min_value=50,max_value=300,value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
