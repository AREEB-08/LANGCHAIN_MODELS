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

parser = StrOutputParser()

# -------------------------
# Prompt 1 → Joke Generator
# -------------------------
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

# -------------------------
# Prompt 2 → Joke Explanation
# -------------------------
prompt2 = PromptTemplate(
    template="Explain this joke in simple terms:\n\n{joke}",
    input_variables=["joke"]
)

explanation_chain = RunnableSequence(prompt2, model, parser)

# -------------------------
# Parallel Chain
# -------------------------
parallel_chain = {
    "joke": RunnablePassthrough(),
    "explanation": explanation_chain
}

# -------------------------
# Final Chain
# -------------------------
final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

# -------------------------
# Run
# -------------------------
result = final_chain.invoke({"topic": "AI"})

print(result, "\n")

print("Joke:\n", result["joke"], "\n")
print("Explanation:\n", result["explanation"], "\n")