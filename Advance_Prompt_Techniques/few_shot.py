
# Few-shot prompting:
# Giving an instruction directly to the model with providing examples.

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

Examples:
Q: can you explain the a+b whole sqaure
Q: can you answer releated to the addition and multiplication rules
Q: give the exaplian of datatypes in python
A: sorry i can give only answers releated to the maths questions only


Always follow exactly this output structure:

Question: <question>
Answer: <final answer>
Explanation: <short explanation>
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
            "content": "Explain rules of additions and multiplication in maths"
        }
    ]
)

print(response.choices[0].message.content)

# with few shot prompting we can bind the output quality with prompt