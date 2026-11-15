"""FastAPI inference API for ML model serving."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Inference API",
    description="Production ML model inference endpoint managed by MLOps Agent",
    version="1.0.0",
)

# --- Models ---
class PredictionRequest(BaseModel):
    features: list[list[float]]
    model_name: str = "iris_classifier"
    model_version: str = "latest"

class PredictionResponse(BaseModel):
    predictions: list[Any]
    model_name: str
    model_version: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None
    version: str

# --- State ---
MODEL_NAME = os.getenv("MODEL_NAME", "iris_classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

# Simulated model state (in production, load real model from MLflow artifact store)
_model_loaded = True

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes probes."""
    return HealthResponse(
        status="healthy" if _model_loaded else "unhealthy",
        model_loaded=_model_loaded,
        model_name=MODEL_NAME,
        version=MODEL_VERSION,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run inference on input features."""
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    # Simulated prediction (in production, use loaded model)
    # For iris classifier: predict class based on simple heuristic
    predictions = []
    for features in request.features:
        if len(features) != 4:
            raise HTTPException(status_code=400, detail=f"Expected 4 features, got {len(features)}")
        # Simple rule-based mock prediction
        if features[2] < 2.5:
            predictions.append({"class": "setosa", "confidence": 0.95})
        elif features[2] < 4.8:
            predictions.append({"class": "versicolor", "confidence": 0.82})
        else:
            predictions.append({"class": "virginica", "confidence": 0.88})
    
    elapsed = (time.time() - start) * 1000
    
    return PredictionResponse(
        predictions=predictions,
        model_name=request.model_name,
        model_version=request.model_version,
        inference_time_ms=round(elapsed, 2),
    )

@app.get("/model/info")
async def model_info():
    """Get information about the currently loaded model."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_loaded": _model_loaded,
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

@app.get("/")
async def root():
    return {"message": "ML Inference API", "docs": "/docs"}
