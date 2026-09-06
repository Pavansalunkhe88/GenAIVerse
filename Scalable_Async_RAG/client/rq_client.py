from redis import Redis
from rq import Queue

queque = Queue(connection=Redis(
    host="vector-db", 
    port=6379, db=0
))

queque.enqueue("worker.process_query", "What is the capital of France?")