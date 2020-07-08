import numpy as np
import pandas as pd
import os

files = os.listdir()
keys = ("initial_stock_price", "strike_price", "maturity", "interest_rate", "dividend_yield",
  "volatility", "rate_of_mean_reversion", "correlation_of_stock_variance", "long_term_variance", "volatility_of_variance",
  "type","knock_type", "barrier_type", "barrier_price", "trigger_price_1", "trigger_price_2", "lookback_type")
for data_file in files:
    if not data_file.endswith(".npy"):
        continue
    res = 0
    print(data_file +"'s shape")
    with open(data_file, "rb") as f:
        res = np.load(f)
        while True:
            try:
                data = np.load(f)
            except:
                if len(res.shape) > 1 and res.shape[1] == 17:
                    df = pd.DataFrame(data=res, columns=keys)
                    df.to_csv("res.csv", index=None)
                    print(df)
                break
            res = np.concatenate((res, data))