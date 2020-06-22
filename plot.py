from pricer import GBM, GBMSA
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    step_num = 1000
    # model = GBM(initial_stock_price=100, strike_price=110, maturity=1,
    # interest_rate=0.2, dividend_yield=0.1, volatility=0.2)

    model = GBMSA(initial_stock_price=100, strike_price=110, maturity=1,
    interest_rate=0.2, dividend_yield=0.1, rate_of_mean_reversion=1,
    correlation_of_stock_variance=-0.4, long_term_variance=0.01,
    volatility_of_variance=0.02, initial_variance=0.01)

    paths = model.stock_path(path_num=100, step_num=step_num)
    times = np.linspace(0, int(model.T), step_num)

    for i in paths:
        plt.plot(times, i)
    plt.title("Simulation of Movement of Asset Price")
    plt.xlabel(r"Times ($year^{-1}$)")
    plt.ylabel("Asset Price")
    plt.show()
