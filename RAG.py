import streamlit as st
from openai import OpenAI
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


st.title("RAG")

with st.sidebar:
    openAI_API_KEY = st.text_input("OPENAI API KEY", key = "chatbot_api_key", type = "password")
    uploadedFile = st.file_uploader("File uploader");

def ragGeneration(uploadedFile, openAI_API_KEY, prompt):
    if uploadedFile is not None:
        docs = [uploadedFile.read().decode()]
        text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
        splits = text_splitter.create_documents(docs)
        vectorStore = Chroma.from_documents(splits, embedding = OpenAIEmbeddings(openai_api_key = openAI_API_KEY))
        retriever = vectorStore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm = OpenAI(openai_api_key = openAI_API_KEY), chain_type = 'stuff', retriever = retriever)
        return qa.run(prompt)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role" : "assistant", "content" : "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openAI_API_KEY:
        st.info("Please enter OpenAI API key to continue.")
        st.stop()
    if uploadedFile is not None:
        with st.spinner("Calculating..."):
            response = ragGeneration(uploadedFile, openAI_API_KEY, prompt)
            st.session_state.messages.append({"role": "assistant", "content" : "response"})
            st.chat_message("assistant").write(response)
    else:
        client = OpenAI(api_key = openAI_API_KEY)
        st.session_state.messages.append({"role":"user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model = "gpt-4o-mini", messages = st.session_state.messages)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": "msg"})
        st.chat_message("assistant").write(msg)
