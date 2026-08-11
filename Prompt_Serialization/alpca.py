# If you mean ALPACA prompting, it refers to the instruction-following format popularized by the Alpaca instruction-tuning work from Stanford.

# Basic format
# ### Instruction:
# Explain inheritance in Python.

# ### Input:
# Use a simple example suitable for a beginner.

# ### Response:
# Inheritance is an OOP mechanism...

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

instruction = "Explain inheritance in Python."

input_data = """
The learner is a beginner.
Use simple language and provide one practical example.
"""

prompt = f"""
### Instruction:
{instruction}

### Input:
{input_data}

### Response:
"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response.choices[0].message.content)