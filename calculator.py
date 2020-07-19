import numpy as np


def EU_Monte_Carlo(model, paths):
    """
    @param model:
    @param paths:
    @return:
    """
    price = 0
    if model.option_type == "call":
        price = paths[:, -1] - model.K
    elif model.option_type == "put":
        price = model.K - paths[:, -1]
    price = np.where(price > 0, price, 0)
    return np.exp(-model.r * model.T) * np.mean(price)


def AM_Monte_Carlo(model, paths):
    """
    @param paths:
    @param model: Model containing the option parameters
    @return:
    """
    price = np.zeros_like(paths)
    step_num = price.shape[1]
    delta_t = model.T / (step_num-1)
    if model.option_type == "call":
        price = paths - model.K
        price = np.where(price > 0, price, 0)
    elif model.option_type == "put":
        price = model.K - paths
        price = np.where(price > 0, price, 0)
    for idx in range(2, step_num):
        back_val = np.exp(-model.r * delta_t) * price[:, -(idx - 1)]
        rg = np.polyfit(paths[:, -idx], back_val, deg=2)
        hold_val = np.polyval(rg, paths[:, -idx])
        price[:, -idx] = np.where(hold_val > price[:, -idx], back_val, price[:, -idx])
    return np.exp(-model.r * delta_t)*np.mean(price[:, 1])


def barrier_Monte_Carlo(model, paths):
    price = 0
    step_num = paths.shape[1]
    if model.knock_type == "in":
        check = np.full(paths.shape[0], False)
        if model.barrier_type == "up":
            for i in range(step_num):
                check[np.where(paths[:, i] >= model.barrier_price)] = True
        else:
            for i in range(step_num):
                check[np.where(paths[:, i] <= model.barrier_price)] = True
        if model.option_type == "call":
            payoff = paths[:, -1] - model.K
        else:
            payoff = model.K - paths[:, -1]
        payoff = np.where(payoff > 0, payoff, 0)
        price = np.where(check, payoff, 0)
    else:
        check = np.full(paths.shape[0], True)
        if model.barrier_type == "up":
            for i in range(step_num):
                check[np.where(paths[:, i] >= model.barrier_price)] = False
        else:
            for i in range(step_num):
                check[np.where(paths[:, i] <= model.barrier_price)] = False
        if model.option_type == "call":
            payoff = paths[:, -1] - model.K
        else:
            payoff = model.K - paths[:, -1]
        payoff = np.where(payoff > 0, payoff, 0)
        price = np.where(check, payoff, 0)
    price *= np.exp(-model.r * model.T)
    return np.mean(price) 


def gap_Monte_Carlo(model, paths):
    res = []
    for path in paths:
        temp = path
        if model.option_type == 'call':
            if model.X2 >= model.X1:
                if temp[-1] > model.X2:
                    res.append(temp[-1] - model.X1)
                else:
                    res.append(0)
            if model.X2 < model.X1:
                if model.X2 < path[-1] < model.X1:
                    res.append(model.X2 - model.X1)
                else:
                    res.append(0)
        elif model.option_type == 'put':
            if model.X2 < model.X1:
                if temp[-1] < model.X2:
                    res.append(model.X1 - temp[-1])
                else:
                    res.append(0)
            if model.X2 > model.X1:
                if model.X2 < path[-1] and model.X1 < path[-1]:
                    res.append(model.X2 - model.X1)
                else:
                    res.append(0)
    return np.exp(-model.r * model.T) * np.mean(res)


def lookback_Monte_Carlo(model, paths):
    max_price = np.max(paths, axis=1)
    min_price = np.min(paths, axis=1)
    profit = 0
    if model.option_type == "call":
        profit = np.where(max_price-model.K>0, max_price-model.K, 0)
    else:
        profit = np.where(model.K-min_price>0, model.K-min_price, 0)
    return np.mean(profit*np.exp(-model.r * model.T))
    