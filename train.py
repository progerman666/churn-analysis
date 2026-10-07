import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
import os

# 1. Загрузка данных
DATA_PATH = "data/telco_churn.csv"
if not os.path.exists(DATA_PATH):
    print("Датасет не найден, генерируем синтетический...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Churn'] = y
    df.to_csv(DATA_PATH, index=False)
    print(f"Сгенерирован датасет и сохранён в {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Загружено {df.shape[0]} записей, {df.shape[1]} признаков")

# 2. Предобработка
# Удаляем колонку с ID, если есть
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Отделяем целевую переменную
if 'Churn' not in df.columns:
    raise ValueError("В датасете нет колонки Churn")
y = df['Churn']
X = df.drop('Churn', axis=1)

# Кодируем категориальные признаки (one-hot)
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Категориальные признаки: {list(categorical_cols)}")
X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# Сохраняем список всех колонок после one-hot
feature_columns = list(X.columns)
print(f"Всего признаков после кодирования: {len(feature_columns)}")

# Разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Обучение моделей
models = {}

# Логистическая регрессия
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
models['LogisticRegression'] = {
    'model': lr,
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'precision': precision_score(y_test, y_pred_lr),
    'recall': recall_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'roc_auc': roc_auc_score(y_test, y_prob_lr)
}

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
models['XGBoost'] = {
    'model': xgb_model,
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'precision': precision_score(y_test, y_pred_xgb),
    'recall': recall_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'roc_auc': roc_auc_score(y_test, y_prob_xgb)
}

# Нейросеть (PyTorch)
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

input_dim = X_train_scaled.shape[1]
nn_model = SimpleNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Обучение
epochs = 50
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = nn_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Оценка
nn_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_prob_nn = nn_model(X_test_tensor).numpy().flatten()
    y_pred_nn = (y_prob_nn >= 0.5).astype(int)
models['NeuralNetwork'] = {
    'model': nn_model,
    'accuracy': accuracy_score(y_test, y_pred_nn),
    'precision': precision_score(y_test, y_pred_nn),
    'recall': recall_score(y_test, y_pred_nn),
    'f1': f1_score(y_test, y_pred_nn),
    'roc_auc': roc_auc_score(y_test, y_prob_nn)
}

# 4. Вывод метрик
print("\nМетрики:")
for name, metrics in models.items():
    print(f"{name}: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics['roc_auc']:.4f}")

# 5. Выбор лучшей модели по ROC-AUC
best_name = max(models, key=lambda k: models[k]['roc_auc'])
best_model = models[best_name]['model']
print(f"\nЛучшая модель: {best_name} (ROC-AUC = {models[best_name]['roc_auc']:.4f})")

# 6. Сохранение модели и scaler
os.makedirs('models', exist_ok=True)

# Всегда сохраняем XGBoost для API (поддерживает SHAP TreeExplainer)
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print("XGBoost сохранён как models/xgboost_model.pkl")

# Сохраняем лучшую модель отдельно (для сравнения)
if best_name == 'XGBoost':
    joblib.dump(best_model, 'models/xgboost_model.pkl')
elif best_name == 'LogisticRegression':
    joblib.dump(best_model, 'models/logistic_regression.pkl')
else:
    torch.save(best_model.state_dict(), 'models/nn_model.pth')

joblib.dump(scaler, 'models/scaler.pkl')
print("Модель и scaler сохранены в папку models/")

# 7. Сохранение списка признаков
with open('models/features.json', 'w') as f:
    json.dump(feature_columns, f)
print("Список признаков сохранён в models/features.json")

# 8. Сохранение метрик
all_metrics = {name: {k: v for k, v in metrics.items() if k != 'model'} for name, metrics in models.items()}
with open('models/all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Сохраняем метрики лучшей модели отдельно
best_metrics = {k: v for k, v in models[best_name].items() if k != 'model'}
with open(f'models/{best_name.lower()}_metrics.json', 'w') as f:
    json.dump(best_metrics, f, indent=2)

print("Метрики сохранены в models/")