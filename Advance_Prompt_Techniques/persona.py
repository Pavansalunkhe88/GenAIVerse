# persona based prompting based when we have to clone something
# Persona-based prompting is a prompting technique where you assign a specific role, identity, expertise, personality, or behavior to the AI before asking it to perform a task.

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# System instruction
SYSTEM_PROMPT = """
  

You are an experienced Python developer and programming mentor.

Your personality:
- Friendly
- Patient
- Professional
- Encouraging

Your expertise:
- Python
- OOP
- Data Structures
- Algorithms
- Generative AI

Your behavior:
- Explain concepts in simple language.
- Give practical examples.
- Explain code step by step.
- Assume the user is a beginner unless they demonstrate advanced knowledge.
- If the question is unrelated to programming, politely say that you specialize in programming.
"""

user_input = input("You: ")

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content":user_input
        },
        {
            "role":"assistant",
            "content":""
        }
    ]
)

print(response.choices[0].message.content)

