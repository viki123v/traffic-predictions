from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.classifiers import WeightedAccidentsClassifier
import joblib
import numpy as np
import os
from pathlib import Path

app = FastAPI(title="Road Danger Prediction API")

templates = Jinja2Templates(directory="api/templates")

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

danger_index_model = joblib.load("models/Wi_regressor.joblib")
classification_model = joblib.load("models/IRAP_classifier.joblib")

wi_model = danger_index_model['model']
wi_scaler = danger_index_model['scaler']
wi_features = danger_index_model['features']
class RoadSegmentInput(BaseModel):
    input_string: str

class PredictionOutput(BaseModel):
    danger_index: float
    is_dangerous: bool
    danger_category: str
    parsed_values: dict
    color: str = "orange"

@app.get("/")
def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    """Check if models are loaded and API is healthy"""
    models_loaded = danger_index_model is not None and classification_model is not None
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_danger(input_data: RoadSegmentInput):
    """
    Predict road segment danger level from formatted string input
    Expected format: LENGTH SECTION; PCI; LIMIT; K.Int. P.T; K.Bridges; PGDS_AVG; Max_Snow; Аve_Height; Ave_Temp; Max_Temp; Ave_Inc; Ave_Rain
    """
    if wi_model is None or classification_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Check server logs.")

    try:
        input_string = input_data.input_string.strip()

        parts = [part.strip() for part in input_string.split(';')]

        if len(parts) != 12:
            raise ValueError(
                f"Expected 12 values separated by semicolons, got {len(parts)}. Please check your input format.")

        try:
            values = [float(part) for part in parts]
        except ValueError as e:
            raise ValueError(f"All values must be numbers. Error: {str(e)}")

        input_dict = {
            'LENGTH SECTION': values[0],
            'PCI': values[1],
            'LIMIT': values[2],
            'K.Int. P.T': values[3],
            'K.Bridges': values[4],
            'PGDS_AVG': values[5],
            'Max_Snow': values[6],
            'Аve_Height': values[7],
            'Ave_Temp': values[8],
            'Max_Temp': values[9],
            'Ave_Inc': values[10],
            'Ave_Rain': values[11]
        }

        X_wi = np.array([input_dict[feat] for feat in wi_features]).reshape(1, -1)

        X_wi_scaled = wi_scaler.transform(X_wi)

        weighted_index = wi_model.predict(X_wi_scaled)[0]

        X_irap = np.array([[weighted_index]])

        irap_class = str(classification_model.predict(X_irap)[0])

        safety_mapping = {
            "Yellow": {"description": "Good safety", "is_dangerous": False, "color": "yellow"},
            "Green": {"description": "Very safe road segment", "is_dangerous": False, "color": "green"},
            "Orange": {"description": "Acceptable safety", "is_dangerous": False, "color": "orange"},
            "Reed": {"description": "Low safety", "is_dangerous": True, "color": "red"},
            "Black": {"description": "Dangerous road segment", "is_dangerous": True, "color": "black"}
        }

        safety_info = safety_mapping.get(irap_class, {
            "description": f"Unknown classification: {irap_class}",
            "is_dangerous": False,
            "color": "orange"
        })

        return PredictionOutput(
            danger_index=float(weighted_index),
            is_dangerous=safety_info["is_dangerous"],
            danger_category=safety_info["description"],
            parsed_values=input_dict,
            color=safety_info["color"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# To run this application:
# 1. Make sure you have templates/index.html
# 2. Make sure your .joblib files are in the same folder as main.py
# 3. Run: uvicorn main:app --reload
# 4. Open browser: http://localhost:8000