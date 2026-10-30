import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HH Salary Advisor API")

MODEL_PATH = "model.bin"

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

class ResumeRequest(BaseModel):
    age: int
    gender: str
    desired_job: str
    experience_months: int
    city: str
    education: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_salary(data: ResumeRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    try:
        input_data = pd.DataFrame([{
            "age": int(data.age),
            "gender": str(data.gender),
            "desired_job": str(data.desired_job),
            "experience_months": int(data.experience_months),
            "city": str(data.city),
            "education": str(data.education)
        }])

        cat_cols = ["gender", "desired_job", "city", "education"]
        for col in cat_cols:
            input_data[col] = input_data[col].astype("category")

        predicted_value = float(model.predict(input_data)[0])
        
        recommended = max(30000, int(predicted_value))
        min_salary = int(recommended * 0.85)
        max_salary = int(recommended * 1.15)

        return {
            "recommended_salary": recommended,
            "min_salary": min_salary,
            "max_salary": max_salary
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        
        # Проверяем, связана ли ошибка с неизвестной категорией XGBoost
        if "category not in the training set" in error_msg:
            raise HTTPException(status_code=400, detail="position not found")
            
        raise HTTPException(status_code=400, detail=error_msg)