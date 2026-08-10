from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    
    messages=[
        {
                "role":"system",
                "content":"Your an expert in maths and only ans releated to maths if question releated to other than maths say sorry to answer the questions"
        },
        {
            "role": "user",
            "content": "multiplication of 2333 to 34"
        }
    ]
)

print(response.choices[0].message.content)