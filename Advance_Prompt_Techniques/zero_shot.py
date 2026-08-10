# Zero-shot prompting:
# Giving an instruction directly to the model without providing examples.

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
You are an expert in mathematics.
Only answer questions related to mathematics.
If the question is not related to mathematics, say:
"Sorry, I can only answer mathematics-related questions."
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "What is the multiplication of 2333 and 34?"
        }
    ]
)

print(response.choices[0].message.content)