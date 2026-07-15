from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------------
# Model
# -------------------------
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free",
    temperature=0.7,
)

prompt=PromptTemplate(
    template="write a summary on this {text}",
    input_variables=["text"]
)


parser = StrOutputParser()




loader = TextLoader(
    r"D:/code floders/PYTHON-5/LANGCHAIN_MODELS/RAG/spam.txt",
    encoding="utf-8"
)

docs = loader.load()
# #print(docs)
# print(type(docs))

# print(docs[0])
chain=prompt |model | parser

print(chain.invoke(docs[0]))