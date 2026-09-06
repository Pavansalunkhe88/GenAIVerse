from fastapi import FastAPI
from client.rq_client import queque

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Async RAG API is running"}


@app.post("/query")
def query(question: str):
    job = queque.enqueue(
        "worker.process_query",
        question
    )

    return {
        "job_id": job.id,
        "status": "queued"
    }