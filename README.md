$content = @"
# Customer Churn Prediction API

Веб-сервис для прогнозирования оттока клиентов на основе машинного обучения.

## Возможности
- Прогноз вероятности оттока для одного клиента (`POST /predict`)
- Пакетный прогноз (`POST /predict/batch`)
- Сравнение трёх моделей (Logistic Regression, XGBoost, Neural Network)
- SHAP-объяснения для каждого прогноза
- Веб-интерфейс для ввода данных и визуализации результатов

## Технологии
- Python 3.11
- FastAPI
- XGBoost
- PyTorch
- SHAP
- Docker

## Быстрый старт

### Локально (без Docker)
1. Установите Python 3.11.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt