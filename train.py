import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def main():
    print("Loading dataset...")
    data_path = "data/dst-3.0_16_1_hh_database.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    # ИСПРАВЛЕНИЕ: Читаем с разделителем ';' и пропускаем битые строки, если они есть
    df = pd.read_csv(data_path, sep=';', on_bad_lines='skip')
    print(f"Original dataset shape: {df.shape}")

    # Очищаем строки без указанной зарплаты
    df = df.dropna(subset=["ЗП"])
    
    # Конвертируем зарплату в числовой формат
    df["ЗП_сум"] = df["ЗП"].astype(str).str.findall(r'(\d+)').apply(lambda x: int(''.join(x)) if x else 0)
    df = df[df["ЗП_сум"] > 0]

    # Извлекаем возраст из колонки 'Пол, возраст'
    if "Пол, возраст" in df.columns:
        df["Возраст"] = df["Пол, возраст"].str.extract(r'(\d+)\s+год').astype(float)

    X = df[["Возраст"]].fillna(30)
    y = df["ЗП_сум"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Сохраняем модель в корень проекта
    with open("model.bin", "wb") as f_out:
        pickle.dump(model, f_out)

    print("Model successfully trained and saved as 'model.bin'")

if __name__ == "__main__":
    main()