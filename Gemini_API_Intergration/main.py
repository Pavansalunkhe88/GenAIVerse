from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6L6XGIhDFEj1h4vLwe6mQP5iGaGDok4HaGQSTSGYxCKPQ"
)

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain how AI works in a few words"
)

print(interaction.output_text)