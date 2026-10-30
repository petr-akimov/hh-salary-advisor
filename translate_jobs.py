import os
import pandas as pd
from deep_translator import GoogleTranslator # опционально, либо используем словарный метод / транслитерацию

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "dst-3.0_16_1_hh_database.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "dst-3.0_16_1_hh_database_eng.csv")

# Базовый словарь точных соответствий для частых профессий (можно дополнять)
EXACT_TRANSLATIONS = {
    "системный администратор": "system administrator",
    "системный администратор windows": "windows system administrator",
    "сетевой инженер": "network engineer",
    "программист": "programmer",
    "разработчик": "developer",
    "инженер-программист": "software engineer",
    "тестировщик": "qa engineer",
    "qa engineer": "qa engineer",
    "москва": "moscow",
}

def translate_position(pos):
    if pd.isna(pos):
        return "specialist"
    
    clean_pos = str(pos).strip().lower()
    
    # 1. Проверяем точное совпадение в словаре
    if clean_pos in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[clean_pos]
    
    # 2. Базовые подмены ключевых слов внутри длинных названий
    translated = clean_pos
    replacements = {
        "системный администратор": "system administrator",
        "инженер": "engineer",
        "разработчик": "developer",
        "руководитель": "head",
        "директор": "director",
        "менеджер": "manager",
        "аналитик": "analyst",
        "тестировщик": "qa engineer",
        "специалист": "specialist",
        "администратор": "administrator",
        "главный": "senior",
        "помощник": "junior",
        "младший": "junior",
        "старший": "senior"
    }
    
    for ru, en in replacements.items():
        translated = translated.replace(ru, en)
        
    return translated.capitalize()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Dataset not found at {INPUT_FILE}")
        return

    print("Loading dataset (this may take a few seconds due to file size)...")
    df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)

    col_name = "Ищет работу на должность:"
    if col_name not in df.columns:
        print(f"Error: Column '{col_name}' not found in dataset.")
        return

    # Вывод списка всех уникальных позиций
    print("\n--- Unique Job Titles Found in Dataset ---")
    unique_positions = df[col_name].dropna().unique()
    print(f"Total unique positions: {len(unique_positions)}")
    
    # Выведем первые 50 для ознакомления
    for i, pos in enumerate(list(unique_positions)[:50]):
        print(f"{i+1}. {pos}")
    if len(unique_positions) > 50:
        print(f"... and {len(unique_positions) - 50} more positions.")

    print("\nTranslating positions to English...")
    df[col_name] = df[col_name].apply(translate_position)

    print(f"Saving translated dataset to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, sep=";", index=False)
    print("Done! Translation completed successfully.")

if __name__ == "__main__":
    main()