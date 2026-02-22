"""
app.py — FastAPI service for football player market value prediction.

Endpoints:
- GET /health: Health check
- POST /predict: Predict player market value from features
  
The service loads a pre-trained RandomForest regressor and preprocessor.
Input features include: age, position, league, career trajectory metrics.
Output: Predicted market value (EUR) and per-request latency.
"""

import joblib
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Football Transfer Market Value Predictor")

# Load model artifact at startup
model_artifact = None


def load_model():
    """Load model and preprocessor from disk."""
    global model_artifact
    try:
        model_artifact = joblib.load("model.joblib")
        print("Model loaded successfully!")
        print(f"  n_estimators: {model_artifact['n_estimators']}")
        print(f"  Test R2: {model_artifact['test_r2']:.4f}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup():
    """Load model on app startup."""
    load_model()


# Request/Response schemas
class PlayerFeatures(BaseModel):
    """
    Input features for market value prediction.
    
    Example features (from football transfer dataset):
    - age: Player age in years
    - career_span_years: Years between first and last valuation
    - years_to_peak: Years from first value to peak
    - value_cagr: Compound annual growth rate of market value
    - value_to_peak_cagr: CAGR from first value to peak value
    - value_multiplier_x: Peak value / first value ratio
    - post_peak_decline_pct: Percentage decline from peak to current
    - value_volatility: Standard deviation of valuations (instability)
    - mean_yoy_growth_rate: Average year-over-year growth
    - num_valuation_points: Count of historical valuations
    - num_clubs_career: Number of clubs played for
    - position_group: Forward/Midfielder/Defender/Goalkeeper (categorical)
    - league_name: Premier League/La Liga/Serie A/etc. (categorical)
    - position: Specific position like "Centre-Forward" (categorical)
    - trajectory: Career arc (rising_star/growing/stable/declining/falling_sharply)
    """
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Output prediction and latency metadata."""
    predicted_value_eur: float
    predicted_value_millions: float
    request_latency_ms: float
    confidence_interval: Dict[str, float]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_artifact is not None,
        "model_type": "RandomForestRegressor",
        "task": "Football player market value prediction"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PlayerFeatures):
    """
    Predict player market value from features.
    
    Args:
        request: PlayerFeatures containing input feature dictionary
        
    Returns:
        PredictionResponse with predicted value and latency
    """
    if model_artifact is None:
        return {"error": "Model not loaded"}
    
    start_time = time.time()
    
    try:
        # Extract features from request
        features_dict = request.features
        
        # Create DataFrame with single sample matching expected columns
        # This should match the columns used in training
        feature_df = pd.DataFrame([features_dict])
        
        # Get numeric and categorical columns
        numeric_features = model_artifact['numeric_features']
        categorical_features = model_artifact['categorical_features']
        
        # Ensure all expected columns exist
        for col in numeric_features + categorical_features:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default fill for missing numeric
        
        # Select only the features used in training
        feature_df = feature_df[numeric_features + categorical_features]
        
        # Preprocess features using the saved preprocessor
        preprocessor = model_artifact['preprocessor']
        X_processed = preprocessor.transform(feature_df)
        
        # Make prediction
        model = model_artifact['model']
        predicted_value = model.predict(X_processed)[0]
        
        # Ensure non-negative prediction
        predicted_value = max(0, predicted_value)
        
        # Estimate confidence interval (simple approach: +/- std of training residuals)
        # Using rough estimate based on model characteristics
        confidence_margin = predicted_value * 0.15  # 15% confidence band
        
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_value_eur=float(predicted_value),
            predicted_value_millions=float(predicted_value / 1_000_000),
            request_latency_ms=latency_ms,
            confidence_interval={
                "lower_eur": max(0, float(predicted_value - confidence_margin)),
                "upper_eur": float(predicted_value + confidence_margin)
            }
        )
    
    except Exception as e:
        return {
            "error": str(e),
            "request_latency_ms": (time.time() - start_time) * 1000
        }


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with model info."""
    if model_artifact is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    
    return {
        "status": "healthy",
        "model_info": {
            "type": "RandomForestRegressor",
            "n_estimators": model_artifact['n_estimators'],
            "test_r2": model_artifact['test_r2'],
            "input_features": {
                "numeric_count": len(model_artifact['numeric_features']),
                "categorical_count": len(model_artifact['categorical_features']),
                "total": len(model_artifact['numeric_features']) + len(model_artifact['categorical_features'])
            }
        },
        "task": "Predict football player market value (EUR)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
