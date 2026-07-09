from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableBranch
)
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------------
# Model
# -------------------------
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.7,
)

parser = StrOutputParser()

# -------------------------
# Prompt 1 → Report Generator
# -------------------------
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

# -------------------------
# Prompt 2 → Summarizer
# -------------------------
prompt2 = PromptTemplate(
    template="Summarize the following text:\n\n{text}",
    input_variables=["text"]
)

summarizer_chain = RunnableSequence(
    {"text": RunnablePassthrough()},  # map string → {"text": string}
    prompt2,
    model,
    parser
)

# -------------------------
# Conditional Branch
# -------------------------
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 500,  # condition
        summarizer_chain               # if True
    ),
    RunnablePassthrough()              # if False
)

# -------------------------
# Final Chain
# -------------------------
final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)

# -------------------------
# Run
# -------------------------
result = final_chain.invoke({"topic": "Russia vs Ukraine war"})

print(result)