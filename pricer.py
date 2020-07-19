import numpy as np
import ghalton
from scipy.stats import norm
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
        step_num = int(self.T * 360)
        paths = np.zeros((path_num, step_num))
        paths[:, 0] = self.S0
        delta_t = self.T / step_num
        halton = ghalton.GeneralizedHalton(step_num-1, 65)
        Z = norm.ppf(halton.get(path_num))
        for i in range(1, step_num):
            paths[:, i] = paths[:, i - 1] * (1+self.r*delta_t+self.sigma*np.sqrt(delta_t)*Z[:, i-1])
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

    def stock_path(self, seed=0, path_num=1000):
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
        step_num = int(self.T * 360)
        delta_t = 1 / 360
        shape = (path_num, step_num)
        halton = ghalton.GeneralizedHalton(path_num, seed)
        w1 = np.sqrt(delta_t)*norm.ppf(halton.get(step_num)).T
        w2 = np.sqrt(delta_t)*norm.ppf(halton.get(step_num)).T
        path = np.zeros(shape)
        path[:, 0] = st
        for i in range(1, step_num):
            st = st + self.r*st*delta_t + np.sqrt(vt)*st*(self.rho*w1[:, i]+np.sqrt(1-self.rho**2)*w2[:, i])
            vt = vt + self.kappa*(self.theta-vt)*delta_t + self.sigma*np.sqrt(vt)*w1[:, i]
            st = np.where(st < 0, 0, st)
            vt = np.where(vt < 0, 0, vt)
            path[:, i] = st
        return path


class GBM_EU(GBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num):
        d1 = (np.log(self.S0/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        if self.option_type == "call":
            return self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == "put":
            return self.K*np.exp(-self.r * self.T)*norm.cdf(-d2) - self.S0*np.exp(-self.q*self.T)*norm.cdf(-d1)


class GBM_AM(GBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num):
        steps = 100
        u = np.exp(self.sigma*np.sqrt(self.T/steps))
        d = 1/u
        P = (np.exp(self.r*self.T/steps)-d)/(u-d)
        prices = np.zeros(steps + 1)
        c_values = np.zeros(steps + 1)
        prices[0]= self.S0*d**steps
        if self.option_type == "call":
            c_values[0]= max(prices[0]-self.K,0)
            for i in range(1, steps+1):
                prices[i] = prices[i-1]*(u**2)
                c_values[i] = max(prices[i]-self.K,0)
            for j in range(steps, 0, -1):
                for i in range(0,j):
                    prices[i]=prices[i+1]*d
                    c_values[i] = max((P*c_values[i+1]+(1-P)*c_values[i])/np.exp(self.r*self.T/steps),prices[i]-self.K)
        elif self.option_type == "put":
            c_values[0] = max(self.K-prices[0],0)
            for i in range(1, steps+1):
                prices[i] = prices[i-1]*(u**2)
                c_values[i] = max(self.K-prices[i],0)
            for j in range(steps, 0, -1):
                for i in range(0,j):
                    prices[i]=prices[i+1]*d
                    c_values[i] = max((P*c_values[i+1]+(1-P)*c_values[i])/np.exp(self.r*self.T/steps), self.K-prices[i])
        return c_values[0]


class GBM_barrier(GBM):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        super().__init__(**kwargs)
        self.knock_type = knock_type
        self.barrier_type = barrier_type
        self.barrier_price = barrier_price

    def get(self, path_num=1000):
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(barrier_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


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
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(gap_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


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
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(lookback_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


class GBMSA_EU(GBMSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(EU_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


class GBMSA_AM(GBMSA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, path_num=1000):
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(AM_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


class GBMSA_barrier(GBMSA):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        super().__init__(**kwargs)
        self.knock_type = knock_type
        self.barrier_type = barrier_type
        self.barrier_price = barrier_price

    def get(self, path_num=1000):
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(barrier_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


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
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(gap_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)


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
        res = []
        seed = 0
        while path_num > 0:
            if path_num > 1000:
                paths = super().stock_path(seed, 1000)
            else:
                paths = super().stock_path(seed, path_num)
            res.append(lookback_Monte_Carlo(self, paths))
            path_num -= 1000
            seed += 1
        return np.mean(res)
