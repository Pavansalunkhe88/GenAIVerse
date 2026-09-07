from dotenv import load_dotenv
import os

from pymongo import MongoClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import Annotated
from typing_extensions import TypedDict


load_dotenv()


# =====================================
# LLM
# =====================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# =====================================
# STATE
# =====================================

class ChatState(TypedDict):

    messages: Annotated[
        list,
        add_messages
    ]


# =====================================
# CHATBOT NODE
# =====================================

def chatbot(state: ChatState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# =====================================
# GRAPH
# =====================================

builder = StateGraph(ChatState)

builder.add_node(
    "chatbot",
    chatbot
)

builder.add_edge(
    START,
    "chatbot"
)

builder.add_edge(
    "chatbot",
    END
)


# =====================================
# MONGODB
# =====================================

MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)

checkpointer = MongoDBSaver(
    client=client,
    db_name="learning_chatbot",
    checkpoint_collection_name="checkpoints",
    writes_collection_name="checkpoint_writes"
)


# =====================================
# COMPILE
# =====================================

app = builder.compile(
    checkpointer=checkpointer
)


# =====================================
# THREAD ID
# =====================================

thread_id = "student_001"

config = {
    "configurable": {
        "thread_id": thread_id
    }
}



# CHAT LOOP


while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config=config
    )

    print("\nAI:", result["messages"][-1].content)