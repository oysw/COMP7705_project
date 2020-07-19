from pricer import *
from calculator import *
import numpy as np

gbm_model = GBM_lookback(
    initial_stock_price=100, 
    strike_price=95, 
    maturity=182/360,
    interest_rate=0.05,
    dividend_yield=0,
    option_type="put",
    volatility=0.2,
    # knock_type="in",
    # barrier_type="up",
    # barrier_price=97,
    lookback_type="floating"
    )

from net import MonteCarloOptionPricing
model = MonteCarloOptionPricing(
    gbm_model.r, 
    gbm_model.S0,
    gbm_model.K,
    gbm_model.T,
    gbm_model.r,
    gbm_model.sigma,
    no_of_slices=gbm_model.T*360,
    simulation_rounds=50000,
    fix_random_seed=True
    )
model.stock_price_simulation()
print(model.LookBackEuropean(gbm_model.option_type))
print(lookback_Monte_Carlo(gbm_model, model.price_array))
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