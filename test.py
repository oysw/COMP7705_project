from pricer import *
from calculator import *
import numpy as np
import ghalton
from scipy.stats import norm
import multiprocessing


gbm_model = GBM_AM(
    initial_stock_price=100, 
    strike_price=110, 
    maturity=182/365,
    interest_rate=0.02,
    dividend_yield=0,
    option_type="put",
    volatility=0.1,
    # knock_type="in",
    # barrier_type="down",
    # barrier_price=97,
    )

paths = gbm_model.stock_path(10000)
res = AM_Monte_Carlo(gbm_model, paths)
print(res)
# from net import MonteCarloOptionPricing
# model = MonteCarloOptionPricing(
#     gbm_model.r, 
#     gbm_model.S0,
#     gbm_model.K,
#     gbm_model.T,
#     gbm_model.r,
#     gbm_model.sigma,
#     no_of_slices=gbm_model.T*360
#     )
# model.stock_price_simulation()
# print(model.american_option_monte_carlo(option_type="put"))

print(gbm_model.get(30000))
# gbmsa_model = GBMSA_AM(
#     initial_stock_price=100, 
#     strike_price=110, 
#     maturity=456/365,
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