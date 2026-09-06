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
@app.get("/job_status/{job_id}")
def get_job_status(job_id: str):
    job = queque.fetch_job(job_id)
    if job is None:
        return {"error": "Job not found"}
    
    return {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.result
    }
