<<<<<<< HEAD
import numpy as np
import pricer
import copy


def dataloader(option, sample_size, path_num=1000, step_num=1000, **kwargs):
    model = 0
    param = dict()
    config = dict(**kwargs)
    # base parameter
    # range : 0.8 -> 1.2
    moneyness = 0.8 + np.random.rand(sample_size) * (1.2 - 0.8)
    param["initial_stock_price"] = np.linspace(100, 100, sample_size)
    param["strike_price"] = param["initial_stock_price"] / moneyness
    # range: 1 day -> 3 year
    param["maturity"] = 1 / 365 + np.random.rand(sample_size) * (3 - 1 / 365)
    # range: 1% -> 3%
    param["interest_rate"] = 0.01 + np.random.rand(sample_size) * (0.03 - 0.01)
    # range: 0% -> 3%
    param["dividend_yield"] = np.random.rand(sample_size) * 0.03
    # GBM parameter
    if option.split("_")[0] == "GBM":
        # range: 0.05 -> 0.5
        param["volatility"] = 0.05 + np.random.rand(sample_size) * (0.5 - 0.05)
    # GBMSA parameter
    if option.split("_")[0] == "GBMSA":
        # range: 0.20 -> 2.00
        param["rate_of_mean_reversion"] = 0.2 + np.random.rand(sample_size) * (2 - 0.2)
        # range: -0.90 -> -0.10
        param["correlation_of_stock_variance"] = -0.9 + np.random.rand(sample_size) * (-0.1 + 0.9)
        # 0.01 -> 0.20
        param["long_term_variance"] = 0.01 + np.random.rand(sample_size) * (0.2 - 0.01)
        # range: 0.05 -> 0.50
        param["volatility_of_variance"] = 0.05 + np.random.rand(sample_size) * (0.5 - 0.05)
        # range: 0.01 -> 0.20
        param["initial_variance"] = 0.01 + np.random.rand(sample_size) * (0.2 - 0.01)
    target = []
    features = []
    var_param = np.array(list(param.values()))
    keys = list(param.keys())
    if config:
        keys.extend(list(config.keys()))
        for i in range(sample_size):
            feature = list(var_param[:, i])
            features.append(copy.copy(feature))
            feature.extend(list(config.values()))
            init_param = dict(zip(keys, feature))
            if hasattr(pricer, option):
                model = getattr(pricer, option)(**init_param)
            if model != 0:
                target.append(model.get(path_num, step_num))
    else:
        for i in range(sample_size):
            feature = list(var_param[:, i])
            features.append(feature)
            init_param = dict(zip(keys, feature))
            if hasattr(pricer, option):
                model = getattr(pricer, option)(**init_param)
            if model != 0:
                target.append(model.get(path_num, step_num))
    return features, target


if __name__ == "__main__":
    data = [
        dataloader("GBM_EU", 100, path_num=2000, step_num=3000, option_type="call"),
        dataloader("GBM_AM", 100, path_num=2000, step_num=3000, option_type="call"),
        dataloader("GBM_barrier", 100, path_num=2000, step_num=3000, option_type="call", knock_type="out",
                   barrier_type="down", barrier_price=90),
        dataloader("GBMSA_EU", 100, path_num=2000, step_num=3000, option_type="call"),
        dataloader("GBMSA_AM", 100, path_num=2000, step_num=3000, option_type="call"),
        dataloader("GBMSA_barrier", 100, path_num=2000, step_num=3000, option_type="call", knock_type="out",
                   barrier_type="down", barrier_price=90),
        dataloader("GBM_gap", 100, path_num=2000, step_num=3000, option_type="call",
                   trigger_price_1=120, trigger_price_2=110),
        dataloader("GBM_lookback", 100, path_num=2000, step_num=3000, option_type="call",
                   lookback_type="floating")
    ]
    for i in data:
        print(i[0], i[1])
=======
import numpy as np
import pricer
import copy


def dataloader(option, option_type, sample_size, **kwargs):
    model = 0
    param = dict()
    config = dict(**kwargs)
    # base parameter
    # range : 0.8 -> 1.2
    moneyness = 0.8 + np.random.rand(sample_size)*(1.2-0.8)
    param["initial_stock_price"] = np.linspace(100, 100, sample_size)
    param["strike_price"] = param["initial_stock_price"] / moneyness
    # range: 1 day -> 3 year
    param["maturity"] = 1/365 + np.random.rand(sample_size)*(3 - 1/365)
    # range: 1% -> 3%
    param["interest_rate"] = 0.01 + np.random.rand(sample_size)*(0.03 - 0.01)
    # range: 0% -> 3%
    param["dividend_yield"] = np.random.rand(sample_size)*0.03
    # GBM parameter
    if option.split("_")[0] == "GBM":
        # range: 0.05 -> 0.5
        param["volatility"] = 0.05 + np.random.rand(sample_size)*(0.5-0.05)
    # GBMSA parameter
    if option.split("_")[0] == "GBMSA":
        # range: 0.20 -> 2.00
        param["rate_of_mean_reversion"] = 0.2 + np.random.rand(sample_size)*(2-0.2)
        # range: -0.90 -> -0.10
        param["correlation_of_stock_variance"] = -0.9 + np.random.rand(sample_size)*(-0.1+0.9)
        # 0.01 -> 0.20
        param["long_term_variance"] = 0.01 + np.random.rand(sample_size)*(0.2-0.01)
        # range: 0.05 -> 0.50
        param["volatility_of_variance"] = 0.05 + np.random.rand(sample_size)*(0.5-0.05)
        # range: 0.01 -> 0.20
        param["initial_variance"] = 0.01 + np.random.rand(sample_size)*(0.2-0.01)
    target = []
    features = []
    var_param = np.array(list(param.values()))
    keys = list(param.keys())
    if config:
        keys.extend(list(config.keys()))
        for i in range(sample_size):
            feature = list(var_param[:, i])
            features.append(copy.copy(feature))
            feature.extend(list(config.values()))
            init_param = dict(zip(keys, feature))
            if hasattr(pricer, option):
                model = getattr(pricer, option)(**init_param)
            if model != 0:
                target.append(model.get(option_type, path_num=1000, step_num=3000))
    else:
        for i in range(sample_size):
            feature = list(var_param[:, i])
            features.append(feature)
            init_param = dict(zip(keys, feature))
            if hasattr(pricer, option):
                model = getattr(pricer, option)(**init_param)
            if model != 0:
                target.append(model.get(option_type, path_num=1000, step_num=3000))
    return features, target


if __name__ == "__main__":
    # features, target = dataloader("GBMSA_barrier", "call", 100, knock_type="out", barrier_type="down", barrier_price=90)
    # features, target = dataloader("GBM_gap", "call", 100, trigger_price_1=110, trigger_price_2=120)
    features, target = dataloader("GBM_lookback", "floating lookback call", 100)
    # features, target = dataloader("GBMSA_barrier", "call", sample_size=100)
    print(features)
    print(target)
    
>>>>>>> fc4deffd81807122d92a8903c1d57f4bd1c3da3c
