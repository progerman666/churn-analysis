import joblib
import os

def load_all_models():
    """Загружает все обученные модели и scaler."""
    models = {}
    # XGBoost
    if os.path.exists("models/xgboost_model.pkl"):
        models["XGBoost"] = joblib.load("models/xgboost_model.pkl")
    # Logistic Regression
    if os.path.exists("models/logistic_regression.pkl"):
        models["Logistic Regression"] = joblib.load("models/logistic_regression.pkl")
    # Scaler
    if os.path.exists("models/scaler.pkl"):
        models["scaler"] = joblib.load("models/scaler.pkl")
    return models

def load_model():
    """Загружает основную модель (XGBoost) и scaler для обратной совместимости."""
    models = load_all_models()
    model = models.get("XGBoost")
    scaler = models.get("scaler")
    return model, scaler