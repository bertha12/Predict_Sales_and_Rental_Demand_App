import json

import joblib

import pandas as pd

import numpy as np

 

MODEL_PATH   = "Models/sales_delta_rf_pipeline.joblib"

COLUMNS_PATH = "Models/sales_delta_rf_columns.json"

 

def predict_sales_price_delta(df_new: pd.DataFrame) -> pd.Series:

    pipe = joblib.load(MODEL_PATH)

 

    cols = pd.read_json(COLUMNS_PATH, typ="series").tolist()

 

    if "LATEST_PRICE" not in df_new.columns:

        raise ValueError("LATEST_PRICE is required in input.")

 

    df = df_new.copy()

 

    # Unit guard: convert millions -> dollars

    try:

        if pd.to_numeric(df["LATEST_PRICE"], errors="coerce").median() < 10:

            df["LATEST_PRICE"] = pd.to_numeric(df["LATEST_PRICE"], errors="coerce") * 1_000_000

    except Exception:

        raise ValueError("Column 'LATEST_PRICE' must be numeric or convertible.")

 

    # Align Δ‑features to schema

    X_delta = df.reindex(columns=cols, fill_value=0)

 

    delta_hat = pipe.predict(X_delta)

 

    sale_hat = df["LATEST_PRICE"].astype(float) + delta_hat

 

    return pd.Series(sale_hat, index=df.index, name="SALE_PRICE_PRED")