from dotenv import load_dotenv
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

#LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# state graph
class ChatState(TypedDict):
    question: str
    answer: str

# node 
def chatbot(state: ChatState):

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }

# create graph
graph = StateGraph(ChatState)


# Add node
graph.add_node("chatbot", chatbot)


# Add edges
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

#complie
app = graph.compile()

# Run chatbot
while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    result = app.invoke({
        "question": question
    })

    print("\nAI:", result["answer"])