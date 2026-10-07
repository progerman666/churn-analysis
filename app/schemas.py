from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerInput(BaseModel):
    tenure: int = Field(..., ge=0, le=72, description="Tenure in months")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    TechSupport: str = Field(..., pattern="^(Yes|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(..., pattern="^(Electronic check|Mailed check|Bank transfer|Credit card)$")
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    top_factors: List[dict]
    model_used: str
    confidence: float

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]