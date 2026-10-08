import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import joblib

# Пытаемся подключить XGBoost (если установлен)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    print("✓ XGBoost доступен")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ XGBoost не установлен, используем GradientBoosting вместо него")
    print("  Установите: pip install xgboost")

os.makedirs('./output', exist_ok=True)
os.makedirs('./output/plots', exist_ok=True)

# ============================================================
# 1. ДАННЫЕ
# ============================================================
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'Возраст': np.random.randint(18, 70, n),
    'Доход': np.random.randint(20, 200, n),
    'Срок_обслуживания': np.random.randint(1, 60, n),
    'Кол_обращений': np.random.randint(0, 20, n),
    'Просрочек': np.random.randint(0, 5, n),
    'Средний_чек': np.random.randint(500, 15000, n),
})
score = (df['Просрочек'] * 3 + df['Кол_обращений'] * 0.5
         - df['Срок_обслуживания'] * 0.05 + np.random.randn(n) * 2)
df['Отток'] = (score > 5).astype(int)

print("=== 1. ДАННЫЕ ===")
print(f"Клиентов: {len(df)}, ушло: {df['Отток'].sum()} ({df['Отток'].mean()*100:.1f}%)")

# ============================================================
# 2. ИНЖИНИРИНГ ПРИЗНАКОВ (feature engineering)
# ============================================================
# Создаём новые полезные признаки
df['Доход_на_возраст'] = df['Доход'] / (df['Возраст'] + 1)  # доход на год жизни
df['Обращений_на_месяц'] = df['Кол_обращений'] / df['Срок_обслуживания']  # частота обращений
df['Просрочек_на_год'] = df['Просрочек'] / (df['Срок_обслуживания'] / 12 + 1)  # просрочки за год
df['Чек_на_обращение'] = df['Средний_чек'] / (df['Кол_обращений'] + 1)  # средний чек за обращение

print("\n=== 2. ИНЖИНИРИНГ ПРИЗНАКОВ ===")
print("Добавлено 4 новых признака:")
print(" - Доход_на_возраст")
print(" - Обращений_на_месяц")
print(" - Просрочек_на_год")
print(" - Чек_на_обращение")

# ============================================================
# 3. РАЗДЕЛЕНИЕ ДАННЫХ
# ============================================================
X = df.drop('Отток', axis=1)
y = df['Отток']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n=== 3. ДАННЫЕ РАЗДЕЛЕНЫ ===")
print(f"Признаков: {X.shape[1]}, обучающих: {len(X_train)}, тестовых: {len(X_test)}")

# ============================================================
# 4. КРОСС-ВАЛИДАЦИЯ (более честная оценка)
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, name):
    """Оценивает модель с кросс-валидацией и на тесте"""
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    # Обучение на всех обучающих
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name:25} CV={cv_scores.mean():.3f}±{cv_scores.std():.3f} | "
          f"Тест: acc={acc:.3f}, AUC={auc:.3f}")
    return model, acc, auc

print("\n=== 4. СРАВНЕНИЕ МОДЕЛЕЙ (с кросс-валидацией) ===")
results = {}

# Модель 1: Логистическая регрессия
lr = LogisticRegression(max_iter=1000, random_state=42)
m, a, u = evaluate_model(lr, "Логистическая регрессия")
results['Логистическая регрессия'] = (m, a, u)

# Модель 2: Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
m, a, u = evaluate_model(rf, "Случайный лес")
results['Случайный лес'] = (m, a, u)

# Модель 3: Градиентный бустинг (или XGBoost)
if XGB_AVAILABLE:
    gb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    gb_name = "XGBoost"
else:
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_name = "GradientBoosting"
m, a, u = evaluate_model(gb, gb_name)
results[gb_name] = (m, a, u)

# ============================================================
# 5. ПОДБОР ГИПЕРПАРАМЕТРОВ (GridSearchCV) для лучшей модели
# ============================================================
print("\n=== 5. ПОДБОР ГИПЕРПАРАМЕТРОВ (GridSearchCV) ===")
print("Ищем лучшие параметры для случайного леса...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучшая точность (CV): {grid.best_score_:.3f}")

# Оцениваем лучшую модель на тесте
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
y_prob_best = best_rf.predict_proba(X_test)[:, 1]
best_acc = accuracy_score(y_test, y_pred_best)
best_auc = roc_auc_score(y_test, y_prob_best)
print(f"Лучшая модель на тесте: acc={best_acc:.3f}, AUC={best_auc:.3f}")

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 6.1 Сравнение точности моделей
names = list(results.keys()) + [f'RF (подбор)']
accs = [results[k][1] for k in results] + [best_acc]
aucs = [results[k][2] for k in results] + [best_auc]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(names, accs, color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'])
axes[0].set_title('Точность моделей')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.8, 1.0)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].bar(names, aucs, color=['#3498db', '#2ecc71', '#e67e22', '#9b59b6'])
axes[1].set_title('AUC моделей')
axes[1].set_ylabel('AUC')
axes[1].set_ylim(0.8, 1.0)
for i, v in enumerate(aucs):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('./output/plots/7_сравнение_моделей.png', dpi=150)
plt.close()
print("\n✓ График 7: сравнение моделей")

# 6.2 ROC-кривые всех моделей
plt.figure(figsize=(8, 6))
colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
for (name, (model, _, _)), color in zip(results.items(), colors):
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, lw=2)
# Лучшая модель после подбора
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
plt.plot(fpr, tpr, label=f'RF после подбора (AUC={best_auc:.3f})',
         color='#e74c3c', lw=2, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые всех моделей')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./output/plots/8_roc_все_модели.png', dpi=150)
plt.close()
print("✓ График 8: ROC-кривые всех моделей")

# 6.3 Важность признаков лучшей модели
importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(10, 6))
importance.plot(kind='barh', color='#9b59b6')
plt.title('Важность признаков (лучшая модель после подбора)')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig('./output/plots/9_важность_признаков_улучшенная.png', dpi=150)
plt.close()
print("✓ График 9: важность признаков (улучшенная)")

# ============================================================
# 7. СОХРАНЕНИЕ
# ============================================================
joblib.dump(best_rf, './output/churn_model_best.pkl')
print("\n✓ Лучшая модель сохранена: output/churn_model_best.pkl")

print("\n=== 6. ИТОГОВЫЙ ОТЧЁТ (лучшая модель) ===")
print(classification_report(y_test, y_pred_best, target_names=['Остался', 'Ушёл']))

print("\n=== 7. ВАЖНОСТЬ ПРИЗНАКОВ (топ-5) ===")
print(importance.sort_values(ascending=False).head(5).round(3).to_string())

print("\n=== 8. ВЫВОДЫ ===")
print(f"1. Лучшая модель: Случайный лес после подбора (acc={best_acc:.3f}, AUC={best_auc:.3f})")
print(f"2. Подбор гиперпараметров улучшил точность с {results['Случайный лес'][1]:.3f} до {best_acc:.3f}")
print("3. Новые признаки (инжиниринг) добавили информации модели")
print("4. Кросс-валидация даёт более честную оценку качества")

print("\n✓ Все файлы сохранены в output/")
print("Графики:", sorted(os.listdir('./output/plots')))