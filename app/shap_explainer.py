import shap
import numpy as np

def get_shap_values(model, X):
    """
    Возвращает SHAP-значения для одного предсказания.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def get_top_factors(model, X, feature_names, top_n=5):
    """
    Возвращает top-N факторов, влияющих на предсказание.
    """
    shap_values = get_shap_values(model, X)
    # Для бинарной классификации берём значения для класса 1 (churn)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = shap_values[0]  # одна строка
    
    # Сортируем по абсолютному значению
    indices = np.argsort(-np.abs(shap_values))[:top_n]
    
    factors = []
    for idx in indices:
        feature = feature_names[idx]
        value = float(shap_values[idx])
        direction = "increases_risk" if value > 0 else "decreases_risk"
        factors.append({
            "feature": feature,
            "shap_value": value,
            "direction": direction
        })
    return factors