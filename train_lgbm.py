import os
import re
import pickle
import pandas as pd
import lightgbm as lgb

DATA_PATH = "data/dst-3.0_16_1_hh_database_eng.csv"
EXCHANGE_RATES_PATH = "data/ExchangeRates.csv" # Если файл в папке data, скорректируйте путь при необходимости
MODEL_PATH = "model.bin"

def parse_gender_age(val):
    gender, age = "М", 30
    if pd.isna(val):
        return gender, age
    parts = str(val).split(",")
    if len(parts) > 0:
        gender = "М" if "Мужчин" in parts[0] or "М" in parts[0] else "Ж"
    if len(parts) > 1:
        match = re.search(r'\d+', parts[1])
        if match:
            age = int(match.group())
    return gender, age

def parse_experience(val):
    if pd.isna(val):
        return 0
    match = re.search(r'Опыт работы (\d+)\n?месяц', str(val))
    if match:
        return int(match.group(1))
    years = re.search(r'(\d+)\s+(лет|года|год)', str(val))
    months = re.search(r'(\d+)\s+(месяцев|месяца|месяц)', str(val))
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m

def parse_city(val):
    if pd.isna(val):
        return "другие"
    city = str(val).split(",")[0].strip()
    if city in ["Москва", "Санкт-Петербург"]:
        return city
    elif city in ["Новосибирск", "Екатеринбург", "Казань", "Нижний Новгород", "Челябинск", "Самара", "Уфа", "Ростов-на-Дону", "Омск", "Красноярск", "Воронеж", "Пермь", "Волгоград", "Краснодар"]:
        return "город-миллионник"
    return "другие"

def parse_education(val):
    if pd.isna(val):
        return "высшее"
    text = str(val).lower()
    if "неоконченное высшее" in text:
        return "неоконченное высшее"
    elif "высшее" in text:
        return "высшее"
    elif "среднее специальное" in text:
        return "среднее специальное"
    return "среднее"

def clean_salary(val):
    if pd.isna(val):
        return None
    match = re.search(r'(\d[\d\s]*\d|\d+)', str(val).replace('\xa0', ''))
    if match:
        try:
            return float(match.group(1).replace(' ', ''))
        except:
            return None
    return None

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, sep=";", low_memory=False)

    # Конвертация валют (если в train.py ранее этого не было, но вы делали это в ноутбуке)
    # Если в train.py конвертация не требуется и вы берете 'ЗП' напрямую, оставьте просто df["target"] = df["ЗП"].apply(clean_salary)
    df["target"] = df["ЗП"].apply(clean_salary)
    df = df.dropna(subset=["target"])

    # Фильтрация выбросов по квантилям (2% и 98%), чтобы убрать аномалии и поднять качество R2
    low_bound = df['target'].quantile(0.02)
    high_bound = df['target'].quantile(0.98)
    df = df[(df['target'] >= low_bound) & (df['target'] <= high_bound)]

    # Извлечение фичей из колонок таблицы
    gender_age = df["Пол, возраст"].apply(parse_gender_age)
    df["gender"] = [g[0] for g in gender_age]
    df["age"] = [g[1] for g in gender_age]

    df["desired_job"] = df["Ищет работу на должность:"].fillna("Специалист")
    df["experience_months"] = df["Опыт работы"].apply(parse_experience)
    df["city"] = df["Город, переезд, командировки"].apply(parse_city)
    df["education"] = df["Образование и ВУЗ"].apply(parse_education)

    features = ["age", "gender", "desired_job", "experience_months", "city", "education"]
    X = df[features].copy()

    # Категориальные столбцы для LightGBM
    cat_cols = ["gender", "desired_job", "city", "education"]
    for col in cat_cols:
        X[col] = X[col].astype("category")

    y = df["target"]

    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()