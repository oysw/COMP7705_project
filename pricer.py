# import ghalton
import numpy as np
from calculator import *


class Pricer:
    def __init__(self, initial_stock_price, strike_price, maturity, interest_rate, dividend_yield, option_type):
        """
        Since the initial stock price and strike price are linear homogeneous, they can be considered as
        one parameter.
        Here, by adding the boundary for each parameter, we make them reasonable.
        @param initial_stock_price:
        @param strike_price:
        moneyness (initial_stock_price/strike_price): 0.8 -> 1.2
        @param maturity: 1 day -> 3 year
        @param interest_rate: 1% -> 3%
        @param dividend_yield: 0% -> 3%
        @param option_type: call, put
        """
        self.S0 = initial_stock_price
        self.K = strike_price
        self.moneyness = self.S0 / self.K
        self.T = maturity
        self.r = interest_rate
        self.q = dividend_yield
        self.option_type = option_type


class GBM(Pricer):
    """
    Geometric Brownian Motion
    """
    def __init__(self, volatility, **kwargs):
        """
        @param volatility: 0.05 -> 0.5
        @param kwargs:
        """
        super().__init__(**kwargs)
        self.sigma = volatility

    def stock_path(self, path_num=1000):
        """
        @param path_num:
        @return: Simulation of the stock price.
        [
            path_1: [step_1, step_2, ......, step_n]
            path_2: [step_1, step_2, ......, step_n]
            ......
            path_num: [step_1, step_2, ......, step_n]
        ]
        """
        step_num = int(self.T * 365)
        shape = (path_num, step_num)
        paths = np.zeros(shape)
        paths[:, 0] = self.S0
        delta_t = self.T / step_num
        Z = np.random.standard_normal(shape)
        for i in range(1, step_num):
            paths[:, i] = paths[:, i - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * delta_t +
                                                   self.sigma * np.sqrt(delta_t) * Z[:, i])
        return paths


class GBMSA(Pricer):
    """
    Geometric Brownian Motion with Stochastic Arrival
    """

    def __init__(self, rate_of_mean_reversion, correlation_of_stock_variance,
                 long_term_variance, volatility_of_variance, initial_variance, **kwargs):
        """
        @param rate_of_mean_reversion: 0.20 -> 2.00
        @param correlation_of_stock_variance: -0.90 -> -0.10
        @param long_term_variance: 0.01 -> 0.20
        @param volatility_of_variance: 0.05 -> 0.50
        @param initial_variance: 0.01 -> 0.20
        @param kwargs:
        """
        super().__init__(**kwargs)
        self.kappa = rate_of_mean_reversion
        self.rho = correlation_of_stock_variance
        self.theta = long_term_variance
        self.sigma = volatility_of_variance
        self.v0 = initial_variance

    def stock_path(self, path_num=1000):
        """
        @param path_num:
        @return: Simulation of the stock price.
        [
            path_1: [step_1, step_2, ......, step_n]
            path_2: [step_1, step_2, ......, step_n]
            ......
            path_num: [step_1, step_2, ......, step_n]
        ]
        """
        st = self.S0
        vt = self.v0
        step_num = int(self.T * 365)
        delta_t = 1 / 365
        shape = (path_num, step_num)
        w1 = np.sqrt(delta_t)*np.random.standard_normal(shape)
        w2 = np.sqrt(delta_t)*np.random.standard_normal(shape)
        path = np.zeros(shape)
        for i in range(step_num):
            st = st + self.r*st*delta_t + np.sqrt(vt)*st*(self.rho*w1[:, i]+np.sqrt(1-self.rho**2)*w2[:, i])
            vt = vt + self.kappa*(self.theta-vt)*delta_t + self.sigma*np.sqrt(vt)*w1[:, i]
            st = np.where(st < 0, 0, st)
            vt = np.where(vt < 0, 0, vt)
            path[:, i] = st
        return path


class GBM_EU(GBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return EU_Monte_Carlo(self, paths)


class GBM_AM(GBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return AM_Monte_Carlo(self, paths)


class GBM_barrier(GBM):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        super().__init__(**kwargs)
        self.knock_type = knock_type
        self.barrier_type = barrier_type
        self.barrier_price = barrier_price

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return barrier_Monte_Carlo(self, paths)


class GBM_gap(GBM):
    """
    Gap option
    """
    def __init__(self, trigger_price_1, trigger_price_2, **kwargs):
        super().__init__(**kwargs)
        self.X1 = trigger_price_1
        self.X2 = trigger_price_2

    def get(self, path_num=1000):
        """
        :return: The value of specified gap option
        """
        paths = super().stock_path(path_num)
        return gap_Monte_Carlo(self, paths)


class GBM_lookback(GBM):
    """
    Lookback option
    """
    def __init__(self, lookback_type, **kwargs):
        super().__init__(**kwargs)
        self.lookback_type = lookback_type

    def get(self, path_num=1000):
        """
        :return: The value of specified gap option
        """
        paths = super().stock_path(path_num)
        return lookback_Monte_Carlo(self, paths)


class GBMSA_EU(GBMSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return EU_Monte_Carlo(self, paths)


class GBMSA_AM(GBMSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return AM_Monte_Carlo(self, paths)


class GBMSA_barrier(GBMSA):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        self.calculator = barrier_Monte_Carlo(knock_type, barrier_type, barrier_price)
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return self.calculator.get(self, paths)


class GBMSA_barrier(GBMSA):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        super().__init__(**kwargs)
        self.knock_type = knock_type
        self.barrier_type = barrier_type
        self.barrier_price = barrier_price

    def get(self, path_num=1000):
        paths = super().stock_path(path_num)
        return barrier_Monte_Carlo(self, paths)


class GBMSA_gap(GBMSA):
    """
    Gap option
    """
    def __init__(self, trigger_price_1, trigger_price_2, **kwargs):
        super().__init__(**kwargs)
        self.X1 = trigger_price_1
        self.X2 = trigger_price_2

    def get(self, path_num=1000):
        """
        :return: The value of specified gap option
        """
        paths = super().stock_path(path_num)
        return gap_Monte_Carlo(self, paths)


class GBMSA_lookback(GBMSA):
    """
    Lookback option
    """
    def __init__(self, lookback_type, **kwargs):
        super().__init__(**kwargs)
        self.lookback_type = lookback_type

    def get(self, path_num=1000):
        """
        :return: The value of specified gap option
        """
        paths = super().stock_path(path_num)
        return lookback_Monte_Carlo(self, paths)
