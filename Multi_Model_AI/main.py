from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

interaction = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_text(
            text="Generate a beautiful caption for this image in exactly 50 words."
        ),
        types.Part.from_uri(
            file_uri="https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d",
            mime_type="image/jpeg"
        )
    ]
)

print(interaction.text)