import os
import pandas as pd

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "dst-3.0_16_1_hh_database.csv")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Dataset not found at {INPUT_FILE}")
        return

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE, sep=";", low_memory=False)

    col_name = "Ищет работу на должность:"
    if col_name not in df.columns:
        print(f"Error: Column '{col_name}' not found in dataset.")
        return

    unique_positions = df[col_name].dropna().unique()
    print(f"\n--- Total unique positions: {len(unique_positions)} ---\n")
    
    for i, pos in enumerate(unique_positions):
        print(f"{i+1}. {pos}")

if __name__ == "__main__":
    main()