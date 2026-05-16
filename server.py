from fastapi import FastAPI
from my_agent import predict

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(event: dict):
    return predict(event)