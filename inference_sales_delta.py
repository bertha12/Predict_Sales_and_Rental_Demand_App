import joblib
import pandas as pd
import numpy as np
import os
 
# Path to the new re-saved model (compatible with sklearn 1.3.2)
MODEL_PATH = os.path.join("Models", "sales_delta_rf_pipeline_132.joblib")
 
def predict_sales_price_delta(df_new: pd.DataFrame) -> pd.Series:
    """
    Loads the RandomForest pipeline and predicts the sales price.
    Input:  df_new (single-row or multi-row dataframe)
    Output: pandas Series of predictions
    """
 
    # Load the model
    pipe = joblib.load(MODEL_PATH)
 
    # Ensure all columns are numeric where expected
    df_new = df_new.copy()
    df_new.replace({True: 1, False: 0}, inplace=True)
 
    # Predict
    delta_hat = pipe.predict(df_new)
 
    return pd.Series(delta_hat)