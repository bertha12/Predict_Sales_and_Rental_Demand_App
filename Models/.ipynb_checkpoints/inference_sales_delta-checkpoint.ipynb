{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b9d19756-c6c8-4213-ba5e-06959f4c1d46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# inference_sales_delta.py\n",
    " \n",
    "import json, joblib\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    " \n",
    "MODEL_PATH   = \"Models/sales_delta_rf_pipeline.joblib\"\n",
    "META_PATH    = \"Models/sales_delta_rf_metadata.json\"\n",
    "COLUMNS_PATH = \"Models/sales_delta_rf_columns.json\"\n",
    " \n",
    "def load_sales_delta_artifacts():\n",
    "    pipe = joblib.load(MODEL_PATH)\n",
    "    with open(META_PATH, \"r\") as f:\n",
    "        meta = json.load(f)\n",
    "    cols = pd.read_json(COLUMNS_PATH, typ=\"series\").tolist()\n",
    "    return pipe, meta, cols\n",
    " \n",
    "def predict_sales_price_delta(df_new: pd.DataFrame) -> pd.Series:\n",
    "    \"\"\"\n",
    "    df_new must include:\n",
    "      - 'LATEST_PRICE' (ask price) — if values are in millions (e.g., 0.75), we convert to dollars.\n",
    "      - Δ-feature columns (we reindex to training schema, filling missing with 0).\n",
    "    Returns a Series 'SALE_PRICE_PRED'.\n",
    "    \"\"\"\n",
    "    pipe, meta, cols = load_sales_delta_artifacts()\n",
    "    assert \"LATEST_PRICE\" in df_new.columns, \"LATEST_PRICE is required.\"\n",
    " \n",
    "    df = df_new.copy()\n",
    " \n",
    "    # Convert to dollars if looks like millions\n",
    "    if df[\"LATEST_PRICE\"].median() < 10:\n",
    "        df[\"LATEST_PRICE\"] = df[\"LATEST_PRICE\"] * 1_000_000\n",
    " \n",
    "    # Align to Δ-feature schema\n",
    "    X_delta_new = df.reindex(columns=cols, fill_value=0)\n",
    " \n",
    "    # Predict Δ and reconstruct SALE\n",
    "    delta_hat = pipe.predict(X_delta_new)\n",
    "    sale_hat  = df[\"LATEST_PRICE\"].astype(float) + delta_hat\n",
    "    return pd.Series(sale_hat, index=df.index, name=\"SALE_PRICE_PRED\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea9a0220-4226-4d7b-bfef-8afa2899e2c1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
