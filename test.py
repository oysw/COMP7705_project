from pricer import GBM_EU, GBM_AM, GBM_barrier, GBMSA_EU
import numpy as np
import ghalton
from scipy.stats import norm

model = GBMSA_EU(
  initial_stock_price=100, 
  strike_price=120, 
  maturity=1700/365,
  interest_rate=0.05,
  dividend_yield=0,
  option_type="call",
  # volatility=0.2,
  # knock_type="in",
  # barrier_type="down",
  # barrier_price=105,
  volatility_of_variance=0.2,
  rate_of_mean_reversion=1,
  correlation_of_stock_variance=-0.5,
  long_term_variance=0.05,
  initial_variance=0.2
  )
print(model.get(1000))
