import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load .env file
load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free",   # or any specific free model
    temperature=0.7,
)

response = llm.invoke([
    HumanMessage(content="Write a short note on indian independence movement")
])

print(response.content)