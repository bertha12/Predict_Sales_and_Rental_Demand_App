import joblib

import pandas as pd

import numpy as np

 

MODEL_PATH   = "Models/rental_rf_tuned_pipeline.joblib"

COLUMNS_PATH = "Models/rental_rf_columns.json"

 

def predict_rental_demand(df_new: pd.DataFrame):

    pipe = joblib.load(MODEL_PATH)

    cols = pd.read_json(COLUMNS_PATH, typ="series").tolist()

 

    df = df_new.copy()

    df = df.reindex(columns=cols, fill_value=0)

 

    prob = pipe.predict_proba(df)[:, 1]

    label = (prob >= 0.50).astype(int)

 

    return pd.Series(prob), pd.Series(label)