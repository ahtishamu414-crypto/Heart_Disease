from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Dict
import joblib
import pandas as pd

app = FastAPI(title="CardioGuard API")

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# Load Models
# =========================

lr = joblib.load("lr_heart.pkl")
svm = joblib.load("svm_heart.pkl")
rf = joblib.load("rf_heart.pkl")
xgb = joblib.load("xgb_heart.pkl")

scaler = joblib.load("scaler_heart.pkl")
features = joblib.load("features_heart.pkl")


# =========================
# Input Schema
# =========================

class PatientData(BaseModel):

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 170,
                "exang": 0,
                "oldpeak": 1.4,
                "slope": 2,
                "ca": 0,
                "thal": 2
            }
        }
    )

    age: float = Field(
        gt=0,
        lt=120,
        description="Patient age"
    )

    sex: Literal[0, 1] = Field(
        description="0 = Female, 1 = Male"
    )

    cp: Literal[0, 1, 2, 3] = Field(
        description="Chest pain type"
    )

    trestbps: float = Field(
        gt=50,
        lt=300,
        description="Resting blood pressure"
    )

    chol: float = Field(
        gt=50,
        lt=700,
        description="Serum cholesterol"
    )

    fbs: Literal[0, 1] = Field(
        description="Fasting blood sugar > 120 mg/dl"
    )

    restecg: Literal[0, 1, 2] = Field(
        description="Resting ECG results"
    )

    thalach: float = Field(
        gt=50,
        lt=250,
        description="Maximum heart rate achieved"
    )

    exang: Literal[0, 1] = Field(
        description="Exercise induced angina"
    )

    oldpeak: float = Field(
        ge=0,
        le=10,
        description="ST depression induced by exercise"
    )

    slope: Literal[0, 1, 2] = Field(
        description="Slope of peak exercise ST segment"
    )

    ca: Literal[0, 1, 2, 3, 4] = Field(
        description="Number of major vessels"
    )

    thal: Literal[0, 1, 2, 3] = Field(
        description="Thalassemia value"
    )


# =========================
# Routes
# =========================

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(data: PatientData):

    d = data.model_dump()

    # =========================
    # Feature Engineering
    # =========================

    d["age_thalach_ratio"] = d["age"] / (d["thalach"] + 1)

    d["chol_age"] = d["chol"] * d["age"]

    d["bp_chol_ratio"] = d["trestbps"] / (d["chol"] + 1)

    # =========================
    # DataFrame Creation
    # =========================

    df_input = pd.DataFrame([d])[features]

    # =========================
    # Scaling
    # =========================

    df_scaled = scaler.transform(df_input)

    # =========================
    # Predictions
    # =========================

    results = {}

    model_configs = [
        ("Logistic Regression", lr, df_scaled),
        ("SVM", svm, df_scaled),
        ("Random Forest", rf, df_input),
        ("XGBoost", xgb, df_input),
    ]

    for name, model, X_in in model_configs:

        prob = float(model.predict_proba(X_in)[0][1])

        results[name] = {
            "probability": round(prob * 100, 2),
            "prediction": int(prob >= 0.5)
        }

    # =========================
    # Ensemble Consensus
    # =========================

    votes = sum(
        1
        for r in results.values()
        if r["prediction"] == 1
    )

    avg_prob = round(
        sum(r["probability"] for r in results.values()) / 4,
        2
    )

    return {
        "models": results,
        "consensus": {
            "votes": votes,
            "avg_probability": avg_prob,
            "high_risk": votes >= 2
        }
    }