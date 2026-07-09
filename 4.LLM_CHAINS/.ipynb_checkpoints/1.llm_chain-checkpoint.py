import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load .env
load_dotenv()

# Create LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free",
    temperature=0.7,
)

# Create Prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a summary for {topic}"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run
topic = input("Enter the topic: ")

output = chain.invoke({"topic": topic})

print(output["text"])