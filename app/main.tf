import os
import pickle
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HH Salary Advisor API")

# Путь к модели в корне проекта
MODEL_PATH = "model.bin"

# Путь к SQLite (если задана переменная окружения через K8s PVC, используем её, иначе локальный файл)
DB_PATH = os.getenv("DATABASE_URL", "sqlite:////data/app.db").replace("sqlite:///", "")

def init_db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER,
                gender TEXT,
                experience_months INTEGER,
                city TEXT,
                education TEXT,
                predicted_salary REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database init warning: {e}")

# Инициализируем БД при старте
init_db()

# Загрузка модели
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

class ResumeRequest(BaseModel):
    age: int
    gender: str
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
        # Формируем признаки так же, как при обучении (в train.py мы использовали только Возраст)
        # Если в будущем добавите больше фичей, расширьте этот массив
        features = [[float(data.age)]]
        
        predicted_value = float(model.predict(features)[0])
        
        # Устанавливаем рыночный диапазон (±15% от предсказания)
        min_salary = int(predicted_value * 0.85)
        max_salary = int(predicted_value * 1.15)
        recommended = int(predicted_value)

        # Сохранение запроса и результата в SQLite
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO predictions (age, gender, experience_months, city, education, predicted_salary) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (data.age, data.gender, data.experience_months, data.city, data.education, recommended)
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"DB save error: {db_err}")

        return {
            "recommended_salary": recommended,
            "min_salary": min_salary,
            "max_salary": max_salary
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))