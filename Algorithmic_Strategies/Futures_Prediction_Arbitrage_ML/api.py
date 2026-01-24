"""
Production REST API for Futures Price Prediction
================================================

FastAPI-based API for real-time predictions and model management.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from datetime import datetime

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.feature_engine import FeatureEngine
from src.utils import load_config, setup_logging, set_random_seeds

# Initialize FastAPI app
app = FastAPI(
    title="Futures Price Prediction API",
    description="ML-based short-term futures price prediction using order book data",
    version="2.0.0"
)

# Load configuration
config = load_config()
logger = setup_logging(config)
set_random_seeds(config)

# Initialize components
data_processor = DataProcessor(config)
feature_engine = FeatureEngine(config)

# Global model storage
models = {}
model_dir = Path(config.get("model_saving", {}).get("directory", "models"))


class OrderBookData(BaseModel):
    """Request model for order book data."""
    timestamp: str
    bid_prices: List[float] = Field(..., min_items=1, description="Bid prices (descending)")
    bid_quantities: List[float] = Field(..., min_items=1, description="Bid quantities")
    ask_prices: List[float] = Field(..., min_items=1, description="Ask prices (ascending)")
    ask_quantities: List[float] = Field(..., min_items=1, description="Ask quantities")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-01-19T10:30:00",
                "bid_prices": [100.5, 100.4, 100.3],
                "bid_quantities": [10, 20, 15],
                "ask_prices": [100.6, 100.7, 100.8],
                "ask_quantities": [15, 10, 20]
            }
        }


class PredictionRequest(BaseModel):
    """Request model for batch predictions."""
    order_book_data: List[OrderBookData]
    model_name: str = Field(default="xgb_regressor", description="Model to use for prediction")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[float]
    timestamp: str
    model_name: str
    confidence: Optional[List[float]] = None


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting Futures Price Prediction API...")
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return
    
    # Load all available models
    for model_file in model_dir.glob("*.pkl"):
        model_name = model_file.stem
        try:
            model = joblib.load(model_file)
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
    
    logger.info(f"API ready with {len(models)} models")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Futures Price Prediction API",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": len(models) > 0
    }


@app.get("/models")
async def list_models():
    """List all available models."""
    model_info = []
    for name in models.keys():
        model_info.append({
            "name": name,
            "type": type(models[name]).__name__,
            "loaded": True
        })
    
    return {"models": model_info, "count": len(model_info)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on order book data.
    
    Args:
        request: Prediction request with order book data
        
    Returns:
        Predictions with metadata
    """
    logger.info(f"Prediction request received for {len(request.order_book_data)} samples")
    
    # Check if model exists
    if request.model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not found. Available: {list(models.keys())}"
        )
    
    try:
        # Convert request data to DataFrame
        data_dicts = [ob.dict() for ob in request.order_book_data]
        df = pd.DataFrame(data_dicts)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate data
        df = data_processor.validate_data(df)
        
        # Engineer features
        df = feature_engine.engineer_features(df)
        
        # Select features used by the model
        if feature_engine.selected_features:
            X = df[feature_engine.selected_features]
        else:
            # Use all numeric features
            X = df.select_dtypes(include=[np.number])
        
        # Fill NaN (from rolling features)
        X = X.fillna(0)
        
        # Make predictions
        model = models[request.model_name]
        predictions = model.predict(X)
        
        # Get confidence if available (for classifiers)
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            confidence = probabilities.max(axis=1).tolist()
        
        response = PredictionResponse(
            predictions=predictions.tolist(),
            timestamp=datetime.now().isoformat(),
            model_name=request.model_name,
            confidence=confidence
        )
        
        logger.info(f"Predictions generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/single")
async def predict_single(order_book: OrderBookData, model_name: str = "xgb_regressor"):
    """
    Make a single prediction.
    
    Args:
        order_book: Single order book snapshot
        model_name: Model to use
        
    Returns:
        Single prediction
    """
    request = PredictionRequest(order_book_data=[order_book], model_name=model_name)
    response = await predict(request)
    
    return {
        "prediction": response.predictions[0],
        "timestamp": response.timestamp,
        "model_name": response.model_name,
        "confidence": response.confidence[0] if response.confidence else None
    }


@app.post("/models/load")
async def load_model(model_name: str):
    """
    Load a specific model from disk.
    
    Args:
        model_name: Name of the model file (without .pkl extension)
    """
    model_path = model_dir / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        models[model_name] = model
        logger.info(f"Model loaded: {model_name}")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' loaded successfully",
            "model_type": type(model).__name__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """
    Unload a model from memory.
    
    Args:
        model_name: Name of the model to unload
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")
    
    del models[model_name]
    logger.info(f"Model unloaded: {model_name}")
    
    return {
        "status": "success",
        "message": f"Model '{model_name}' unloaded"
    }


if __name__ == "__main__":
    import uvicorn
    
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    
    uvicorn.run(app, host=host, port=port)
