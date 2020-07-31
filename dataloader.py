import os
import numpy as np
import pricer
import pandas as pd
import itertools
import multiprocessing
import ghalton
import progressbar

keys = ("option", "type", "initial_stock_price", "strike_price", "maturity", "interest_rate", "dividend_yield",
    "volatility", "rate_of_mean_reversion", "correlation_of_stock_variance", "long_term_variance", "volatility_of_variance",
    "knock_type", "barrier_type", "barrier_price", "trigger_price_1", "trigger_price_2", "lookback_type")

def dataloader(option, random, sample_size, path_num=10000):
    model = 0
    S0 = 100
    random = [random[:, i] for i in range(len(keys))]
    param = dict()
    config = dict.fromkeys(keys, np.array([0 for i in range(sample_size)]))
    # base parameter
    # range : 0.8 -> 1.2
    moneyness = 0.8 + random.pop(0)*(1.2-0.8)
    config["initial_stock_price"] = param["initial_stock_price"] = np.array([S0 for i in range(sample_size)])
    config["strike_price"] = param["strike_price"] = np.round(param["initial_stock_price"] / moneyness, 1)
    # range: 1 day -> 3 year
    config["maturity"] = param["maturity"] = np.round((1 + random.pop(0) * 1080) / 360, 4)
    # range: 1% -> 3%
    config["interest_rate"] = param["interest_rate"] = np.round((1+random.pop(0)*2)/100, 3) 
    # range: 0% -> 3%
    config["dividend_yield"] = param["dividend_yield"] = np.round(random.pop(0)*3/100, 3) 
    # "call" and "put"
    config["type"] = np.round(random.pop(0)+1)
    param["option_type"] = np.where(config["type"]==1, "call", "put")
    # GBM parameter
    if option.split("_")[0] == "GBM":
        # range: 0.05 -> 0.5
        config["volatility"] = param["volatility"] = np.round(0.05+ random.pop(0)*(0.5-0.05), 2)
    # GBMSA parameter
    if option.split("_")[0] == "GBMSA":
        # range: 0.01 -> 0.20
        config["volatility"] = param["initial_variance"] = np.round(0.01+ random.pop(0)*(0.2-0.01), 2)
        # range: 0.20 -> 2.00
        config["rate_of_mean_reversion"] = param["rate_of_mean_reversion"] = np.round(0.2+ random.pop(0)*(2-0.2), 2)
        # range: -0.90 -> -0.10
        config["correlation_of_stock_variance"] = param["correlation_of_stock_variance"] = np.round(-0.9 + random.pop(0)*(-0.1 + 0.9), 2)
        # 0.01 -> 0.20
        config["long_term_variance"] = param["long_term_variance"] = np.round(0.01+ random.pop(0)*(0.2-0.01), 2)
        # range: 0.05 -> 0.50
        config["volatility_of_variance"] = param["volatility_of_variance"] = np.round(0.05+ random.pop(0)*(0.5-0.05), 2)
    if option.split("_")[1] == "EU":
        config["option"] = np.array([1 for i in range(sample_size)])
    if option.split("_")[1] == "AM":
        config["option"] = np.array([0 for i in range(sample_size)])
    if option.split("_")[1] == "barrier":
        config["option"] = np.array([2 for i in range(sample_size)])
        # "in" and "out"
        config["knock_type"] = np.round(random.pop(0)+1)
        param["knock_type"] = np.where(config["knock_type"]==1, "in", "out")
        # "up" and "down"
        config["barrier_type"] = np.round(random.pop(0)+1)
        param["barrier_type"] = np.where(config["barrier_type"]==1, "up", "down")
        # range (1 -> 1.5)*S0 for up, range (0.5 -> 1)*S0 for down.
        factors = np.round(random.pop(0)*0.5, 2)
        factors = np.where(config["type"]==1, 1+factors, 1-factors)
        config["barrier_price"] = param["barrier_price"] = S0*factors
    if option.split("_")[1] == "gap":
        config["option"] = np.array([3 for i in range(sample_size)])
        # range (95% -> 105%)*S0
        factors = np.round(1 + random.pop(0)*0.1 - 0.05, 3)
        config["trigger_price_1"] = param["trigger_price_1"] = S0*factors
        factors = np.round(1 + random.pop(0)*0.1 - 0.05, 3)
        config["trigger_price_2"] = param["trigger_price_2"] = S0*factors
    if option.split("_")[1] == "lookback":
        config["option"] = np.array([4 for i in range(sample_size)])
        # "floating" and "fixed"
        config["lookback_type"] = np.round(random.pop(0)+1)
        param["lookback_type"] = np.where(config["lookback_type"]==1, "floating", "fixed")
    features = np.array(list(config.values())).T.tolist()
    target = []
    df = pd.DataFrame.from_dict(param)
    bar = progressbar.ProgressBar(max_value=sample_size)
    for i in range(sample_size):
        init = df.iloc[i].to_dict()
        if hasattr(pricer, option):
            model = getattr(pricer, option)(**init)
        if model != 0:
            target.append(model.get(path_num))
        bar.update(i+1)
    bar.finish()
    return features, [round(i, 2) for i in target]


def generate(path, amount, batch_nums):
    process_num = multiprocessing.cpu_count()
    asset_type = ["GBM", "GBMSA"]
    option_type = ["EU", "AM", "barrier", "gap", "lookback"]

    if not os.path.exists(path):
        os.makedirs(path)

    batch_size = amount // batch_nums // process_num
    print("Batch size is {}".format(batch_size))
    gen = ghalton.GeneralizedHalton(len(keys), 65)

    for batch in range(batch_nums):
        print("Batch {} begins!".format(batch+1))
        for ass, opt in itertools.product(asset_type, option_type):
            print("Begin generating " + ass + "_"+ opt + " data")
            p = multiprocessing.Pool(process_num)
            result = []
            for i in range(process_num):
                random = np.array(gen.get(batch_size))
                result.append(p.apply_async(func=dataloader, args=(ass + "_" + opt, random, batch_size)))
            p.close()
            p.join()
            features = []
            targets = []
            for res in result:
                feature, target = res.get()
                features.extend(feature)
                targets.extend(target)
            data_df = pd.DataFrame(data=features)
            data_df.to_csv(os.path.join(path, opt[0].upper() + "_" + ass + "_" + "data.csv"), mode="a", index=None, header=False)
            label_df = pd.DataFrame(data=targets)
            label_df.to_csv(os.path.join(path, opt[0].upper() + "_" + ass + "_" + "label.csv"), mode="a", index=None, header=False)
        print("Batch {} ends!".format(batch+1))


if __name__ == "__main__":
    generate("result", 400000, 80)
