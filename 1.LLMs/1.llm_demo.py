from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load .env variables
load_dotenv()

# Initialize latest OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",   # latest recommended model
    temperature=0.7
)

# Generate response
response = llm.invoke("What is the capital of India?")

print(response.content)
