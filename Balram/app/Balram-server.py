import os
from typing import List
from fastapi import FastAPI
from langchain_cohere.chat_models import ChatCohere
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes
from langchain_cohere.chat_models import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere.embeddings import CohereEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.environ['COHERE_API_KEY']

llm = ChatCohere(cohere_api_key=cohere_api_key, temperature=0.3)

# load retriever
import bs4
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

loader = WebBaseLoader(web_paths=[
    'https://en.wikipedia.org/wiki/Pest_control#:~:text=In%20agriculture%2C%20pests%20are%20kept,of%20a%20certain%20pest%20species.',
    'https://cpdonline.co.uk/knowledge-base/food-hygiene/pest-control/#the-laws-around-pest-control'],
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                           class_=('mw-content-ltr mw-parser-output', 'wpb_wrapper')
                       )))
data = loader.load()
text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)

docs = text_splitter.split_documents(data)


# Embedding

embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
vector = FAISS.from_documents(docs, embeddings)
retriever = vector.as_retriever()

base_compressor = CohereRerank()

compression_retriever = ContextualCompressionRetriever(base_compressor=base_compressor, base_retriever=retriever)

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300))
retriver_tool = create_retriever_tool(retriever=compression_retriever, name='Vectored DB',
                                      description='use this for any queries regarding pest control')

tools = [wikipedia, retriver_tool]

memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')

agent_prompt = hub.pull('hwchase17/structured-chat-agent')
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, handle_parsing_errors=True)

app = FastAPI()

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        extra={"widget": {"type": "chat", "input": "location"}},
    )

class Output(BaseModel):
    output: str
        
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)


# @app.post("/agent")
# async def chatbot(input_text: str):
#     chat_history = memory.buffer_as_messages
#     response = agent_executor.invoke({
#         'input': input_text,
#         'chat_history': chat_history
#     })
#     return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9000)
