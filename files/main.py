"""
main.py - FastAPI app that loads model.pkl and serves diabetes predictions.

Run with:
    pip install fastapi uvicorn scikit-learn joblib numpy pandas
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to test it.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import numpy as np

# The model lives in this dict so your routes can reach it after startup.
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO 1: load model once at startup
    ml_model["predictor"] = joblib.load("model.pkl")

    yield

    # TODO 2: clear the model on shutdown
    ml_model.clear()


app = FastAPI(title="Diabetes Risk API", lifespan=lifespan)


# -- Request shape: what the caller SENDS ------------------------------------
class PatientFeatures(BaseModel):
    # TODO 3: 8 fields, same order as training
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# -- Response shape: what you SEND BACK --------------------------------------
class PredictionResult(BaseModel):
    # TODO 4
    prediction: int
    result_label: str
    confidence: float


@app.get("/")
def health():
    return {"status": "running", "model_loaded": "predictor" in ml_model}


@app.post("/predict", response_model=PredictionResult)
def predict(features: PatientFeatures):
    model = ml_model.get("predictor")
    if model is None:
        # TODO 5
        raise HTTPException(status_code=503, detail="Model not loaded")

    # TODO 6: input array in the SAME column order as training
    X = np.array([[
        features.Pregnancies,
        features.Glucose,
        features.BloodPressure,
        features.SkinThickness,
        features.Insulin,
        features.BMI,
        features.DiabetesPedigreeFunction,
        features.Age,
    ]])

    # TODO 7: prediction (0 or 1)
    pred = int(model.predict(X)[0])

    # TODO 8: confidence = probability of the predicted class
    conf = round(float(model.predict_proba(X)[0][pred]), 3)

    # TODO 9: return result
    return PredictionResult(
        prediction=pred,
        result_label="Diabetic" if pred == 1 else "Not Diabetic",
        confidence=conf,
    )
