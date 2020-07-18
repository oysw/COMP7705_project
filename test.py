from pricer import *
from calculator import EU_Monte_Carlo
import numpy as np
import ghalton
from scipy.stats import norm

gbm_model = GBM_EU(initial_stock_price=100, 
    strike_price=120, 
    maturity=456/365,
    interest_rate=0.05,
    dividend_yield=0,
    option_type="put",
    volatility=0.2)
# gbmsa_model = GBMSA_AM(
#     initial_stock_price=100, 
#     strike_price=105, 
#     maturity=182/365,
#     interest_rate=0.05,
#     dividend_yield=0,
#     option_type="call",
#     volatility=0.2,
#     knock_type="in",
#     barrier_type="up",
#     barrier_price=110,
#     trigger_price_1=90,
#     trigger_price_2=120,
#     lookback_type="floating",
#     volatility_of_variance=0,
#     rate_of_mean_reversion=1,
#     correlation_of_stock_variance=0,
#     long_term_variance=0.04,
#     initial_variance=0.04
#     )
path_num = 16400
def calc(path_num):
    res = []
    seed = 0
    while path_num > 0:
        if path_num > 1000:
            paths = gbm_model.stock_path(seed, 1000)
        else:
            paths = gbm_model.stock_path(seed, path_num)
        res.append(EU_Monte_Carlo(gbm_model, paths))
        path_num -= 1000
        seed += 1
    return np.mean(res)
print(calc(path_num))
print(gbm_model.get(16400))
