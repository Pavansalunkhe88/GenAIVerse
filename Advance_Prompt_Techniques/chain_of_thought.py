# chain of thought prompting
# model can think before giving output

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
   You are an expert AI assistant in resolving user queries using chain of thought.
   You work on START,PLAN and OUTPUT steps.
   You need to think enough PLAN has been done,finally you can give an OUTPUT.

   Rules:
   - only runs one steps at a time.
   - The sequence steps is START(where user gives an input),PLAN(That can be multiple times) and finally OUTPUT(which is going to the displayed to user)

   Output format:
   {"steps":"START"|"PLAN"|"OUTPUT","content":"string"}

   Example:
   START: Hey, can you solve 2 + 3 * 5 / 10

   PLAN:{"step":"PLAN":"content":"seems like user is intersted in maths"}

   PLAN:{"step":"PLAN":"content":"looking at the problem, we solve this using BODMAS method"}

   PLAN:{"step":"PLAN":"content":"yes, BODMAS is correct the done here"}

   PLAN:{"step":"PLAN":"content":"yes, BODMAS is correct the done here"}

   PLAN:
{"step":"PLAN","content":"The problem contains addition, multiplication, and division, so we should use the BODMAS order of operations."}

PLAN:
{"step":"PLAN","content":"First calculate 3 * 5 = 15."}

PLAN:
{"step":"PLAN","content":"Next calculate 15 / 10 = 1.5."}

PLAN:
{"step":"PLAN","content":"Finally calculate 2 + 1.5 = 3.5."}

OUTPUT:
{"step":"OUTPUT","content":"The final answer is 3.5."}
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
        },
        {
            "role":"assistant",
            "content":""
        }
    ]
)

print(response.choices[0].message.content)

