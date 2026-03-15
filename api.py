from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")


class Customer(BaseModel):
    age: int
    income: float
    total_spending: float
    num_web_purchases: int
    num_catalog_purchases: int
    num_store_purchases: int
    num_web_visit: int
    recency: int


@app.get("/")
def home():
    return {"message": "Customer Segmentation API is running"}


@app.post("/predict")
def predict(customer: Customer):

    data = np.array([[
        customer.age,
        customer.income,
        customer.total_spending,
        customer.num_web_purchases,
        customer.num_catalog_purchases,
        customer.num_store_purchases,
        customer.num_web_visit,
        customer.recency
    ]])

    scaled = scaler.transform(data)

    cluster = model.predict(scaled)

    return {"cluster": int(cluster[0])}