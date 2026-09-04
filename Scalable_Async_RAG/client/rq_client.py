from redis import Redis
from rq import Queue

queque = Queue(connection=Redis(
    host="vector-db", 
    port=6379, db=0
))

