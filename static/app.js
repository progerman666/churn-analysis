const form = document.getElementById('churn-form');
const resultDiv = document.getElementById('result');
const compareDiv = document.getElementById('compare-result');
const compareBtn = document.getElementById('compare-btn');

// Обновление значения tenure
const tenureSlider = document.getElementById('tenure');
const tenureValue = document.getElementById('tenure-value');
tenureSlider.addEventListener('input', () => {
    tenureValue.textContent = tenureSlider.value;
});

// Функция сбора данных из формы
function getFormData() {
    return {
        tenure: parseInt(tenureSlider.value),
        MonthlyCharges: parseFloat(document.getElementById('monthlycharges').value),
        TotalCharges: parseFloat(document.getElementById('totalcharges').value),
        Contract: document.getElementById('contract').value,
        InternetService: document.getElementById('internetservice').value,
        PhoneService: document.getElementById('phoneservice').value,
        TechSupport: document.getElementById('techsupport').value,
        OnlineSecurity: document.getElementById('onlinesecurity').value,
        OnlineBackup: document.getElementById('onlinebackup').value,
        DeviceProtection: document.getElementById('deviceprotection').value,
        StreamingTV: document.getElementById('streamingtv').value,
        StreamingMovies: document.getElementById('streamingmovies').value,
        PaymentMethod: document.getElementById('paymentmethod').value,
        gender: document.getElementById('gender').value,
        SeniorCitizen: parseInt(document.getElementById('seniorcitizen').value),
        Partner: document.getElementById('partner').value,
        Dependents: document.getElementById('dependents').value,
        MultipleLines: document.getElementById('multiplelines').value
    };
}

// Отправка запроса на /predict
async function predict(data) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
}

// Отображение результата
function displayResult(result) {
    resultDiv.classList.remove('hidden');
    const probability = result.churn_probability;
    const fill = document.getElementById('progress-fill');
    const probText = document.getElementById('probability-text');
    const predictionLabel = document.getElementById('prediction-label');
    const factorsDiv = document.getElementById('factors');
    const modelInfo = document.getElementById('model-info');

    // Прогресс-бар
    fill.style.width = `${probability * 100}%`;
    probText.textContent = `${(probability * 100).toFixed(1)}%`;

    // Цвет
    if (probability < 0.3) {
        fill.style.background = '#a6e3a1';
    } else if (probability < 0.7) {
        fill.style.background = '#f9e2af';
    } else {
        fill.style.background = '#f38ba8';
    }

    // Прогноз
    predictionLabel.textContent = result.churn_prediction ? '⚠️ Клиент склонен к оттоку' : '✅ Клиент лоялен';

    // Факторы
    factorsDiv.innerHTML = '';
    result.top_factors.forEach(factor => {
        const item = document.createElement('div');
        item.className = 'factor-item';
        const direction = factor.direction === 'increases_risk' ? 'increase' : 'decrease';
        const icon = factor.direction === 'increases_risk' ? '↑' : '↓';
        item.innerHTML = `
            <span class="icon ${direction}">${icon}</span>
            <span class="feature">${factor.feature}</span>
            <span class="shap-value">${factor.shap_value.toFixed(3)}</span>
        `;
        factorsDiv.appendChild(item);
    });

    // Модель
    modelInfo.textContent = `Модель: ${result.model_used}`;
}

// Обработчик отправки формы
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = getFormData();
    try {
        const result = await predict(data);
        displayResult(result);
        compareDiv.classList.add('hidden');
    } catch (error) {
        alert('Ошибка при запросе: ' + error.message);
    }
});

// Сравнение моделей
compareBtn.addEventListener('click', async () => {
    const data = getFormData();
    try {
        const response = await fetch('/predict/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const results = await response.json();
        displayCompare(results);
    } catch (error) {
        alert('Ошибка при сравнении: ' + error.message);
    }
});

function displayCompare(results) {
    compareDiv.classList.remove('hidden');
    const tbody = document.querySelector('#compare-table tbody');
    tbody.innerHTML = '';
    results.forEach(modelResult => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${modelResult.model}</td>
            <td>${(modelResult.probability * 100).toFixed(1)}%</td>
            <td>${modelResult.prediction ? 'Да' : 'Нет'}</td>
        `;
        tbody.appendChild(row);
    });
}
