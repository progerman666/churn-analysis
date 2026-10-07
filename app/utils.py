import pandas as pd
import numpy as np
import json
import os

def load_feature_columns():
    """Загружает список признаков из models/features.json"""
    with open('models/features.json', 'r') as f:
        return json.load(f)

def preprocess_input(data: dict, scaler):
    """
    Преобразует входные данные клиента в формат, ожидаемый моделью.
    Возвращает нормализованный массив признаков.
    """
    feature_columns = load_feature_columns()
    
    # Создаём DataFrame с одной строкой
    df = pd.DataFrame([data])
    
    # Кодируем категориальные признаки (one-hot)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Добавляем недостающие колонки (если какие-то категории отсутствуют)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Переупорядочиваем колонки в том же порядке, что и при обучении
    df = df[feature_columns]
    
    # Нормализуем числовые признаки (те, что были масштабированы)
    # В нашем случае мы масштабировали все признаки, поэтому нормализуем все
    df_scaled = scaler.transform(df)
    
    return df_scaled[0]