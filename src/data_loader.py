import os
import pandas as pd

def load_from_dir(path: str) -> dict:
    """
    Load all CSV files inside a folder and return dict(name=dataframe)
    """
    datasets = {}
    files = [f for f in os.listdir(path) if f.endswith(".csv")]

    for f in files:
        try:
            df = pd.read_csv(os.path.join(path, f), low_memory=False)
            datasets[f.replace(".csv","")] = df
            print(f"✔ Loaded {f} → {df.shape}")
        except Exception as e:
            print(f"❌ Failed to load {f}: {e}")

    return datasets
