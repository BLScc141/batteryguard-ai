from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ID_COLUMNS = {"vehicle_id", "battery_serial"}
TARGET = "battery_failure"

def preprocess_file(source: Path, destination: Path) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo {TARGET}")
    if df[TARGET].isna().any():
        raise ValueError("battery_failure contiene nulos; revisar etiquetas antes de entrenar")
    duplicates = int(df.duplicated().sum())
    categorical = [
        c for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in ID_COLUMNS and c != TARGET
    ]
    for col in categorical:
        values = df[col].astype("string").str.strip().replace("", pd.NA)
        df[col] = values.fillna("Desconocido")
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, encoding="utf-8")
    print(f"[INFO] Filas: {len(df):,}; columnas: {len(df.columns)}")
    print(f"[INFO] Duplicados detectados (no eliminados): {duplicates}")
    print(f"[INFO] Columnas categóricas normalizadas: {len(categorical)}")
    print(f"[INFO] Nulos numéricos preservados para imputación dentro del pipeline.")
    print(f"[SUCCESS] Dataset procesado: {destination.relative_to(ROOT).as_posix()}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/ev_battery_failure_prediction_Dataset.csv")
    parser.add_argument("--output", default="data/processed/ev_battery_failure_clean.csv")
    args = parser.parse_args()
    preprocess_file((ROOT / args.input).resolve(), (ROOT / args.output).resolve())

if __name__ == "__main__":
    main()


