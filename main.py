from pydantic import BaseModel
from transformers import pipeline
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer


app=FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

class InputText(BaseModel):
    question:str

@app.get('/')
def home():
    return {"Application is running "}

@app.post('/answer')
def embed(data:InputText):
    embedding = model.encode(data.question).tolist()
    return {'embedding':embedding}