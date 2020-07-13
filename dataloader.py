import os
import numpy as np
import pricer
import pandas as pd
import itertools
import multiprocessing
import ghalton


def dataloader(option, sample_size, path_num=1000):
    model = 0
    S0 = 100
    seed = np.random.randint(100)
    gen = ghalton.GeneralizedHalton(sample_size, seed)
    gen.get(seed)
    keys = ("option", "type", "initial_stock_price", "strike_price", "maturity", "interest_rate", "dividend_yield",
    "volatility", "rate_of_mean_reversion", "correlation_of_stock_variance", "long_term_variance", "volatility_of_variance",
    "knock_type", "barrier_type", "barrier_price", "trigger_price_1", "trigger_price_2", "lookback_type")
    param = dict()
    config = dict.fromkeys(keys, np.array([0 for i in range(sample_size)]))
    # base parameter
    # range : 0.8 -> 1.2
    moneyness = 0.8 + np.array(gen.get(1)).ravel() * (1.2 - 0.8)
    config["initial_stock_price"] = param["initial_stock_price"] = np.linspace(S0, S0, sample_size)
    config["strike_price"] = param["strike_price"] = param["initial_stock_price"] / moneyness
    # range: 1 day -> 3 year
    config["maturity"] = param["maturity"] = 1 / 365 + np.array(gen.get(1)).ravel() * (3 - 1 / 365)
    # range: 1% -> 3%
    config["interest_rate"] = param["interest_rate"] = 0.01 + np.array(gen.get(1)).ravel() * (0.03 - 0.01)
    # range: 0% -> 3%
    config["dividend_yield"] = param["dividend_yield"] = np.array(gen.get(1)).ravel() * 0.03
    # "call" and "put"
    config["type"] = np.array([i for i in (1+np.array(gen.get(1)).ravel()*2).astype(int)])
    param["option_type"] = np.where(config["type"]==1, "call", "put")
    # GBM parameter
    if option.split("_")[0] == "GBM":
        # range: 0.05 -> 0.5
        config["volatility"] = param["volatility"] = 0.05 + np.array(gen.get(1)).ravel() * (0.5 - 0.05)
    # GBMSA parameter
    if option.split("_")[0] == "GBMSA":
        # range: 0.01 -> 0.20
        config["volatility"] = param["initial_variance"] = 0.01 + np.array(gen.get(1)).ravel() * (0.2 - 0.01)
        # range: 0.20 -> 2.00
        config["rate_of_mean_reversion"] = param["rate_of_mean_reversion"] = 0.2 + np.array(gen.get(1)).ravel() * (2 - 0.2)
        # range: -0.90 -> -0.10
        config["correlation_of_stock_variance"] = param["correlation_of_stock_variance"] = -0.9 + np.array(gen.get(1)).ravel() * (-0.1 + 0.9)
        # 0.01 -> 0.20
        config["long_term_variance"] = param["long_term_variance"] = 0.01 + np.array(gen.get(1)).ravel() * (0.2 - 0.01)
        # range: 0.05 -> 0.50
        config["volatility_of_variance"] = param["volatility_of_variance"] = 0.05 + np.array(gen.get(1)).ravel() * (0.5 - 0.05)
    if option.split("_")[1] == "EU":
        config["option"] = np.array([1 for i in range(sample_size)])
    if option.split("_")[1] == "AM":
        config["option"] = np.array([0 for i in range(sample_size)])
    if option.split("_")[1] == "barrier":
        config["option"] = np.array([2 for i in range(sample_size)])
        # "in" and "out"
        config["knock_type"] = np.array([i for i in (1+np.array(gen.get(1)).ravel()*2).astype(int)])
        param["knock_type"] = np.where(config["type"]==1, "in", "out")
        # "up" and "down"
        config["barrier_type"] = np.array([i for i in (1+np.array(gen.get(1)).ravel()*2).astype(int)])
        param["barrier_type"] = np.where(config["type"]==1, "up", "down")
        # range (1 -> 1.5)*S0 for up, range (0.5 -> 1)*S0 for down.
        factors = np.array(gen.get(1)).ravel()*0.5
        factors = np.where(config["type"]==1, 1+factors, 1-factors)
        config["barrier_price"] = param["barrier_price"] = S0*factors
    if option.split("_")[1] == "gap":
        config["option"] = np.array([3 for i in range(sample_size)])
        # range (95% -> 105%)*S0
        factors = 1 + np.array(gen.get(1)).ravel()*0.1 - 0.05
        config["trigger_price_1"] = param["trigger_price_1"] = S0*factors
        factors = 1 + np.array(gen.get(1)).ravel()*0.1 - 0.05
        config["trigger_price_2"] = param["trigger_price_2"] = S0*factors
    if option.split("_")[1] == "lookback":
        config["option"] = np.array([4 for i in range(sample_size)])
        # "floating" and "fixed"
        config["lookback_type"] = np.array([i for i in (1+np.array(gen.get(1)).ravel()*2).astype(int)])
        param["lookback_type"] = np.where(config["type"]==1, "floating", "fixed")
    features = np.array(list(config.values())).T.tolist()
    target = []
    df = pd.DataFrame.from_dict(param)
    for i in range(sample_size):
        init = df.iloc[i].to_dict()
        if hasattr(pricer, option):
            model = getattr(pricer, option)(**init)
        if model != 0:
            target.append(model.get(path_num))
    return features, target


def generate(path, amount):
    process_num = multiprocessing.cpu_count()
    asset_type = ["GBM", "GBMSA"]
    option_type = ["EU", "AM", "barrier", "gap", "lookback"]
    keys = ("option", "type", "initial_stock_price", "strike_price", "maturity", "interest_rate", "dividend_yield",
    "volatility", "rate_of_mean_reversion", "correlation_of_stock_variance", "long_term_variance", "volatility_of_variance",
    "knock_type", "barrier_type", "barrier_price", "trigger_price_1", "trigger_price_2", "lookback_type")
    
    if not os.path.exists(path):
        os.makedirs(path)

    gbmsa_data_path = os.path.join(path, "gbmsa_data.csv")
    gbm_data_path = os.path.join(path, "gbm_data.csv")
    gbmsa_label_path = os.path.join(path, "gbmsa_label.csv")
    gbm_label_path = os.path.join(path, "gbm_label.csv")

    data_df = pd.DataFrame(columns=keys)
    if not os.path.isfile(gbmsa_data_path):
        data_df.to_csv(gbmsa_data_path, index=None)
    if not os.path.isfile(gbm_data_path):
        data_df.to_csv(gbm_data_path, index=None)
    label_df = pd.DataFrame(columns=["value"])
    if not os.path.isfile(gbmsa_label_path):
        label_df.to_csv(gbmsa_label_path, index=None)
    if not os.path.isfile(gbm_label_path):
        label_df.to_csv(gbm_label_path, index=None)

    total_amount = amount
    batch_size = total_amount // process_num

    for ass, opt in itertools.product(asset_type, option_type):
        print("Begin generating " + ass + "_"+ opt + " data")
        p = multiprocessing.Pool(process_num)
        result = []
        for i in range(process_num):
            result.append(p.apply_async(func=dataloader, args=(ass + "_" + opt, batch_size)))
        p.close()
        p.join()
        features = []
        targets = []
        for res in result:
            feature, target = res.get()
            features.extend(feature)
            targets.extend(target)
        if ass == "GBMSA":
            data_df = pd.DataFrame(data=features)
            data_df.to_csv(gbmsa_data_path, mode="a", index=None, header=False)
            label_df = pd.DataFrame(data=targets)
            label_df.to_csv(gbmsa_label_path, mode="a", index=None, header=False)
        if ass == "GBM":
            data_df = pd.DataFrame(data=features)
            data_df.to_csv(gbm_data_path, mode="a", index=None, header=False)
            label_df = pd.DataFrame(data=targets)
            label_df.to_csv(gbm_label_path, mode="a", index=None, header=False)


if __name__ == "__main__":
    generate("result", 16)
