from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda

load_dotenv()

prompt=PromptTemplate(
    template="write a joke on {topic}",
    input_variables=['topic']
)
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free",
    temperature=0.7,
)

parser=StrOutputParser()
chain=RunnableSequence(prompt,model,parser)

print(chain.invoke({"topic":'AI'}))