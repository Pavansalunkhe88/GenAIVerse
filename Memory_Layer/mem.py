import os

from dotenv import load_dotenv
from google import genai
from mem0 import Memory


# ============================================
# 1. Load Environment Variables
# ============================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add it to your .env file."
    )


# ============================================
# 2. Gemini Client
# ============================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================
# 3. Mem0 Configuration
# ============================================

config = {

    "version": "v1.1",

    # ----------------------------------------
    # Gemini Embedding
    # ----------------------------------------

    "embedder": {
        "provider": "gemini",

        "config": {
            "api_key": GEMINI_API_KEY,
            "model": "gemini-embedding-001",
        },
    },

    # ----------------------------------------
    # Gemini LLM
    # ----------------------------------------

    "llm": {
        "provider": "gemini",

        "config": {
            "api_key": GEMINI_API_KEY,
            "model": "gemini-2.5-flash",
        },
    },

    # ----------------------------------------
    # Qdrant Vector Database
    # ----------------------------------------

    "vector_store": {
    "provider": "qdrant",
    "config": {
        "host": "localhost",
        "port": 6333,
        "collection_name": "gemini_memory_768",
    },
},
}


# ============================================
# 4. Create Mem0 Client
# ============================================

mem_client = Memory.from_config(config)


# ============================================
# 5. Chat Function
# ============================================

def chat(user_input, user_id):

    # ----------------------------------------
    # STEP 1: Search Previous Memories
    # ----------------------------------------

    memories = mem_client.search(
        user_input,
        filters={
            "user_id": user_id
        }
    )

    print("\n========== RETRIEVED MEMORIES ==========")

    if memories:
        for memory in memories:
            print("-", memory.get("memory"))

    else:
        print("No previous memories found.")

    print("=========================================\n")


    # ----------------------------------------
    # STEP 2: Create Memory Context
    # ----------------------------------------

    memory_context = ""

    if memories:

        memory_context = "\n".join(
            [
                memory["memory"]
                for memory in memories
                if "memory" in memory
            ]
        )


    # ----------------------------------------
    # STEP 3: Create Gemini Prompt
    # ----------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the following information about the user
when it is relevant to the current question.

USER MEMORY:
{memory_context}

CURRENT USER MESSAGE:
{user_input}

Instructions:

1. Answer the user's question naturally.
2. Use the memory only when relevant.
3. Do not mention the memory system.
4. Do not invent personal information.
5. If the memory does not contain the answer,
   simply answer normally.
"""


    # ----------------------------------------
    # STEP 4: Generate Gemini Response
    # ----------------------------------------

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text


    # ----------------------------------------
    # STEP 5: Save Conversation to Mem0
    # ----------------------------------------

    mem_client.add(

        [
            {
                "role": "user",
                "content": user_input
            },

            {
                "role": "assistant",
                "content": answer
            }
        ],

        user_id=user_id
    )


    # ----------------------------------------
    # STEP 6: Return Answer
    # ----------------------------------------

    return answer


# ============================================
# 6. Main Chat Loop
# ============================================

if __name__ == "__main__":

    user_id = "pavan"

    print("\n===================================")
    print("       Gemini + Mem0 + Qdrant")
    print("===================================")
    print("Type 'exit' or 'quit' to stop.")
    print("===================================\n")


    while True:

        user_input = input("You: ")

        # ------------------------------------
        # Exit condition
        # ------------------------------------

        if user_input.lower().strip() in [
            "exit",
            "quit"
        ]:
            print("\nGoodbye! 👋")
            break


        # ------------------------------------
        # Generate Response
        # ------------------------------------

        try:

            answer = chat(
                user_input,
                user_id
            )

            print("AI:", answer)


        except Exception as e:

            print("\n❌ Error:")
            print(e)

            print("\nPlease check:")
            print("1. Gemini API key")
            print("2. Qdrant is running")
            print("3. Qdrant is available at localhost:6333")