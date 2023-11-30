# QnA bot using llm with streamlit FE.ipynb

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0.5
)

def get_openai_response(question):
    response = llm(question)
    return response

st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")


input = st.text_input("Input: ", key="input")
response = get_openai_response(input)

submit = st.button("Ask")

## if ask button is clicked

if submit:
    st.subheader("The response is")
    st.write(response)