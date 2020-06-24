import ghalton
import numpy as np

from calculator import AM_Monte_Carlo, EU_Monte_Carlo, barrier_Monte_Carlo


class Pricer:
    def __init__(self, initial_stock_price, strike_price, maturity, interest_rate, dividend_yield):
        """
        Since the initial stock price and strike price are linear homogeneous, they can be considered as
        one parameter.
        Here, by adding the boundary for each parameter, we make them reasonable.
        :param moneyness (initial_stock_price/strike_price): 0.8 -> 1.2
        :param maturity: 1 day -> 3 year
        :param interest_rate: 1% -> 3%
        :param dividend_yield: 0% -> 3%
        """
        self.S0 = initial_stock_price
        self.K = strike_price
        self.moneyness = self.S0 / self.K
        self.T = maturity
        self.r = interest_rate
        self.q = dividend_yield


class GBM(Pricer):
    """
    Geometric Brownian Motion
    """

    def __init__(self, volatility, **kwargs):
        super().__init__(**kwargs)
        self.sigma = volatility

    def stock_path(self, path_num=1000, step_num=1000):
        """
        :param path_num:
        :param step_num:
        :return: Matrix with path_num samples (row) and step_num time step (col)
        """
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
        super().__init__(**kwargs)
        self.kappa = rate_of_mean_reversion
        self.rho = correlation_of_stock_variance
        self.theta = long_term_variance
        self.sigma = volatility_of_variance
        self.v0 = initial_variance


    def stock_path(self, path_num=1000, step_num=1000):
        """
        The volatility of this option satisfied some kinds of distribution
        @param path_num:
        @param step_num:
        @return: The stock price through a period of time.
        """
        ln_st = np.log(self.S0)
        ln_vt = np.log(self.v0)
        vt = self.v0
        delta_t = self.T / step_num
        shape = (path_num, step_num)
        es = np.random.standard_normal(shape)
        ev = self.rho * es + np.sqrt(1 - self.rho ** 2) * np.random.standard_normal(shape)
        path = np.zeros(shape)
        for i in range(step_num):
            ln_st = ln_st + (self.r - 0.5 * vt) * delta_t + np.sqrt(vt) * np.sqrt(delta_t) * es[:, i]
            st = np.exp(ln_st)
            ln_vt = ln_vt + (1 / vt) * (self.kappa * (self.theta - vt) - 0.5 * self.sigma ** 2) * delta_t + \
                    self.sigma * (1 / np.sqrt(vt)) * np.sqrt(delta_t) * ev[:, i]
            vt = np.exp(ln_vt)
            path[:, i] = st
        return path


class GBM_EU(GBM):
    """
    Geometric Brownian Motion for European Option
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, option_type, path_num=1000, step_num=1000):
        """
        @param option_type:
        @param path_num:
        @param step_num:
        @return:
        """
        paths = super().stock_path(path_num, step_num)
        return EU_Monte_Carlo(self, paths, option_type)


class GBM_AM(GBM):
    """
    Geometric Brownian Motion for European Option
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, option_type, path_num=1000, step_num=1000):
        """
        @param option_type:
        @param path_num:
        @param step_num:
        @return:
        """
        paths = super().stock_path(path_num, step_num)
        return AM_Monte_Carlo(self, paths, option_type, path_num, step_num)


class GBMSA_EU(GBMSA):
    """
    GBMSA model for European option
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, option_type, path_num=1000, step_num=1000):
        """
        The volatility of this option satisfied some kinds of distribution
        @param path_num:
        @param step_num:
        @param option_type:
        :return: The value of European call/put option.
        """
        paths = super().stock_path(path_num, step_num)
        return EU_Monte_Carlo(self, paths, option_type)


class GBMSA_AM(GBMSA):
    """
    GBMSA model for American option
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, option_type, path_num=1000, step_num=1000):
        """
        :param option_type:
        :param path_num:
        :param step_num:
        :return:
        """
        paths = super().stock_path(path_num, step_num)
        return AM_Monte_Carlo(self, paths, option_type, path_num, step_num)


class GBM_barrier(GBM):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        self.calculator = barrier_Monte_Carlo(knock_type, barrier_type, barrier_price)
        super().__init__(**kwargs)

    def get(self, option_type, path_num=1000, step_num=1000):
        paths = super().stock_path(path_num, step_num)
        return self.calculator.get(self, paths, option_type, path_num, step_num)


class GBMSA_barrier(GBMSA):
    def __init__(self, knock_type, barrier_type, barrier_price, **kwargs):
        self.calculator = barrier_Monte_Carlo(knock_type, barrier_type, barrier_price)
        super().__init__(**kwargs)

    def get(self, option_type, path_num=1000, step_num=1000):
        paths = super().stock_path(path_num, step_num)
        return self.calculator.get(self, paths, option_type, path_num, step_num)


class GBM_gap(GBM):
    """
    Gap option
    """
    def __init__(self, trigger_price_1, trigger_price_2,**kwargs):
        super().__init__(**kwargs)
        self.X1 = trigger_price_1
        self.X2 = trigger_price_2

    def get(self, gap_type, path_num=1000, step_num=1000):
        """
        :return: The value of specified gap option
        """
        paths = super().stock_path(path_num, step_num)
        res = []
        for path in paths:
            temp = path
            if gap_type == 'call':
                if self.X2 >= self.X1:
                    if temp[-1] > self.X2:
                        res.append(temp[-1] - self.X1)
                    else:
                        res.append(0)
                if self.X2 < self.X1:
                    if self.X2 < path[-1] and self.X1 > path[-1]:
                        res.append(self.X2-self.X1)
                    else:
                        res.append(0)
            elif gap_type == 'put':
                if self.X2 < self.X1:
                    if temp[-1] < self.X2:
                        res.append(self.X1-temp[-1])
                    else:
                        res.append(0)
                if self.X2 > self.X1:
                    if self.X2 < path[-1] and self.X1 < path[-1]:
                        res.append(self.X2-self.X1)
                    else:
                        res.append(0)
        return np.exp(-self.r*self.T)*np.mean(res)


class GBM_lookback(GBM):
    """
    Lookback option
    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get(self, lookback_type, path_num=1000, step_num=1000):
        """
        :return: The value of specified gap option
        """
        paths = super().stock_path(path_num, step_num)
        res = []
        for path in paths:
            temp = path
            if lookback_type == 'floating lookback call':
                    res.append(temp[-1] - min(temp))
            if lookback_type == 'floating lookback put':
                    res.append(max(temp)-temp[-1])
            if lookback_type == 'fixed lookback put':
                    res.append(max(temp)-temp[-1])
            if lookback_type == 'fixed lookback call':
                    res.append(max(temp)-temp[-1])
        return np.exp(-self.r*self.T)*np.mean(res)


if __name__ == "__main__":
    # model = GBMSA_barrier(initial_stock_price=100, strike_price=110, maturity=1,
    #                  interest_rate=0.2, dividend_yield=0.1, rate_of_mean_reversion=1,
    #                  correlation_of_stock_variance=-0.4, long_term_variance=0.01,
    #                  volatility_of_variance=0.02, initial_variance=0.01,
    #                  barrier_price=98, barrier_type="down", knock_type="in")

    # model = GBM_EU(initial_stock_price=100, strike_price=110, maturity=1,
    #                interest_rate=0.2, dividend_yield=0.1, volatility=0.2)

    model = GBM_barrier(initial_stock_price=100, strike_price=100, maturity=1,
                    interest_rate=0.2, dividend_yield=0.1, volatility=0.1,
                    barrier_price=98, barrier_type="down", knock_type="in")

    # model = gap(initial_stock_price=100, strike_price=110,trigger_price_1=110, trigger_price_2=120, maturity=1,
    #                  interest_rate=0.2, dividend_yield=0.1, rate_of_mean_reversion=1,
    #                  correlation_of_stock_variance=-0.4, long_term_variance=0.01,
    #                  volatility_of_variance=0.02, initial_variance=0.01)
    # val = model.get("call", 2000, 2000)


    # model = lookback(initial_stock_price=100, strike_price=110, maturity=1,
    #                  interest_rate=0.2, dividend_yield=0.1, volatility=0.1)
    # val = model.get('floating lookback call', 2000, 2000)

    # model = gap(initial_stock_price=100, strike_price=110,trigger_price_1=120, trigger_price_2=110, maturity=1,
    #                  interest_rate=0.2, dividend_yield=0.1, volatility=0.1)
    val = model.get("call", 2000, 2000)
    print(val)

