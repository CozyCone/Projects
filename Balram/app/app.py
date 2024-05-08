import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from collections import deque
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage
from langchain.chains import create_history_aware_retriever
import bs4
import time
from dotenv import load_dotenv

load_dotenv()
cohere_api_key = 'WPaf0QFOOu8TMM5J1WM9Tb3MNPVZQENTp6ymqRXP'

if "vector" not in st.session_state:
    st.session_state.embeddings=CohereEmbeddings()
    loader = WebBaseLoader(web_paths=['https://en.wikipedia.org/wiki/Pest_control#:~:text=In%20agriculture%2C%20pests%20are%20kept,of%20a%20certain%20pest%20species.',
                                  'https://cpdonline.co.uk/knowledge-base/food-hygiene/pest-control/#the-laws-around-pest-control'],
                       bs_kwargs= dict(parse_only = bs4.SoupStrainer(
                           class_ = ('mw-content-ltr mw-parser-output','wpb_wrapper')
                       )))
    st.session_state.loader=loader
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("Balram : A chatbot for farmers")
llm= ChatCohere(cohere_api_key=cohere_api_key,temperature=0.5)

doc_prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, doc_prompt)
base_retriever = st.session_state.vector.as_retriever()
base_compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(base_compressor=base_compressor,base_retriever=base_retriever)
retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
retriver_tool = create_retriever_tool(retriever=compression_retriever,name ='Vectored DB',description='use this for any queries regarding pest control')
wikipedia = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200))
tools = [wikipedia,retriver_tool]

agent_prompt = hub.pull('hwchase17/structured-chat-agent')
agent = create_structured_chat_agent(llm=llm,tools=tools,prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

num_previous_messages = 10
i = 0

text_input = st.chat_input('Enter your prompt here')


while True:

    if text_input:

        start_time = time.process_time()

        if text_input.lower()=='end':
            break

        st.write(f"You : {text_input}")

        chat_history = deque(maxlen=num_previous_messages)

        chat_history.append(HumanMessage(content=text_input))

        response = agent_executor.invoke({
            'chat_history':list(chat_history),
            'input':text_input})
        
        ai_response = response['output']

        chat_history.append(AIMessage(content=ai_response))

        end_time = time.process_time() - start_time

        st.write(f"AI: {ai_response}")

        text_input = st.chat_input('Enter your prompt here')




    