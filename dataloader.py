import numpy as np
import pricer
import copy
import itertools
import multiprocessing


def dataloader(option, sample_size, path_num=1000, **kwargs):
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
                target.append(model.get(path_num))
    else:
        for i in range(sample_size):
            feature = list(var_param[:, i])
            features.append(feature)
            init_param = dict(zip(keys, feature))
            if hasattr(pricer, option):
                model = getattr(pricer, option)(**init_param)
            if model != 0:
                target.append(model.get(path_num))
    return features, target


if __name__ == "__main__":
    process_num = multiprocessing.cpu_count()
    asset_type = ["GBM", "GBMSA"]
    option_type = ["call", "put"]
    option = ["AM", "EU"]
    total_amount = 800
    batch = 1
    batch_size = total_amount // batch // process_num
    print("Batch size is %s" % batch_size)

    for ass, opt, opt_type in itertools.product(asset_type, option, option_type):
        print("Begin generating " + ass + "_"+ opt + " data")
        times = 0
        while times < batch:
            p = multiprocessing.Pool(process_num)
            result = []
            for i in range(process_num):
                result.append(p.apply_async(func=dataloader, args=(ass + "_" + opt, batch_size), kwds={"option_type": opt_type}))
            p.close()
            p.join()
            features = []
            targets = []
            for res in result:
                feature, target = res.get()
                features.extend(feature)
                targets.extend(target)
            features = np.array(features)
            targets = np.array(targets)
            with open(ass + "_" + opt + "_" + opt_type + "_features.npy", "ab") as f:
                np.save(f, features)
            with open(ass + "_" + opt + "_" + opt_type + "_targets.npy", "ab") as f:
                np.save(f, targets)
            times += 1
            print("Batch %s has finished!" % times)
