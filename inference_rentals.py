import joblib
import pandas as pd
import numpy as np
import os
 
MODEL_PATH = os.path.join("Models", "rental_rf_tuned_pipeline_132.joblib")
 
def predict_rental_demand(df_new: pd.DataFrame) -> pd.Series:
    """
    Loads the re-saved RandomForest rental demand model (sklearn 1.3.2 compatible)
    and returns predictions.
    """
    pipe = joblib.load(MODEL_PATH)
 
    df_new = df_new.copy()
    df_new.replace({True: 1, False: 0}, inplace=True)
 
    preds = pipe.predict(df_new)
 
    return pd.Series(preds)