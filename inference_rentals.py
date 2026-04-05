import joblib

import pandas as pd

 

MODEL_PATH = "Models/rental_rf_tuned_pipeline.joblib"

COLUMNS_PATH = "Models/rental_rf_columns.json"

 

def predict_rental_demand(df_new: pd.DataFrame):

    pipe = joblib.load(MODEL_PATH)

    cols = pd.read_json(COLUMNS_PATH, typ="series").tolist()

 

    df = df_new.reindex(columns=cols, fill_value=0)

 

    prob = pipe.predict_proba(df)[:, 1]

    label = (prob >= 0.50).astype(int)

 

    return (

        pd.Series(prob, index=df.index, name="HIGH_DEMAND_PROB"),

        pd.Series(label, index=df.index, name="HIGH_DEMAND_LABEL"),

    )