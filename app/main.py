from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

from app.model import load_all_models
from app.schemas import CustomerInput, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.utils import preprocess_input
from app.shap_explainer import get_top_factors

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Загрузка моделей при старте
models = load_all_models()
scaler = models.get("scaler")  # scaler общий
# Основная модель для /predict — XGBoost
model = models.get("XGBoost")

# Загрузка списка признаков
with open('models/features.json', 'r') as f:
    FEATURE_NAMES = json.load(f)

@app.get("/", include_in_schema=False)
def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="XGBoost model not loaded")
        X = preprocess_input(customer.dict(), scaler)
        X = X.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        pred = bool(prob >= 0.5)
        top_factors = get_top_factors(model, X, FEATURE_NAMES, top_n=5)
        return PredictionResponse(
            churn_probability=float(prob),
            churn_prediction=pred,
            top_factors=top_factors,
            model_used="XGBoost",
            confidence=float(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="XGBoost model not loaded")
        predictions = []
        for customer in batch.customers:
            X = preprocess_input(customer.dict(), scaler)
            X = X.reshape(1, -1)
            prob = model.predict_proba(X)[0][1]
            pred = bool(prob >= 0.5)
            top_factors = get_top_factors(model, X, FEATURE_NAMES, top_n=5)
            predictions.append(PredictionResponse(
                churn_probability=float(prob),
                churn_prediction=pred,
                top_factors=top_factors,
                model_used="XGBoost",
                confidence=float(prob)
            ))
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/compare")
def predict_compare(customer: CustomerInput):
    """Сравнение всех доступных моделей."""
    try:
        results = []
        for name, m in models.items():
            if name == "scaler":
                continue
            X = preprocess_input(customer.dict(), scaler)
            X = X.reshape(1, -1)
            prob = m.predict_proba(X)[0][1]
            results.append({
                "model": name,
                "probability": float(prob),
                "prediction": bool(prob >= 0.5)
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def get_models():
    with open("models/all_metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics