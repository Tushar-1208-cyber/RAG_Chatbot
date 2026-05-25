from groq import Groq

# 🔑 API client
import os

api_key = os.getenv("GROQ_API_KEY")


# 🔹 LLM 1 (Fast - LLaMA)
def get_llm1(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# 🔹 LLM 2 (More powerful - Mixtral)
def get_llm2(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content