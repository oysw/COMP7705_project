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
    delta_t = model.T / step_num
    if model.option_type == "call":
        tmp = paths[:, -1] - model.K
        price[:, -1] = np.where(tmp > 0, tmp, 0)
    elif model.option_type == "put":
        tmp = model.K - paths[:, -1]
        price[:, -1] = np.where(tmp > 0, tmp, 0)
    for idx in range(2, step_num + 1):
        back_val = np.exp(-model.r * delta_t) * price[:, -(idx - 1)]
        c_val = np.zeros_like(back_val)
        if model.option_type == "call":
            c_val = paths[:, -idx] - model.K
        elif model.option_type == "put":
            c_val = model.K - paths[:, -idx]
        price[:, -idx] = np.where(back_val > c_val, back_val, c_val)
    return np.mean(price[:, 0])


def barrier_Monte_Carlo(model, paths):
    price = 0
    if model.knock_type == "in":
        if model.barrier_type == "up":
            res = []
            for path in paths:
                if path[np.where(path >= model.barrier_price)].size == 0:
                    res.append(0)
                else:
                    if model.option_type == "call":
                        res.append(max(path[-1] - model.K, 0))
                    elif model.option_type == "put":
                        res.append(max(model.K - path[-1], 0))
            if res:
                price = np.mean(res)
        elif model.barrier_type == "down":
            res = []
            for path in paths:
                if path[np.where(path <= model.barrier_price)].size == 0:
                    res.append(0)
                else:
                    if model.option_type == "call":
                        res.append(max(path[-1] - model.K, 0))
                    elif model.option_type == "put":
                        res.append(max(model.K - path[-1], 0))
            if res:
                price = np.mean(res)
    elif model.knock_type == "out":
        if model.barrier_type == "up":
            if model.S0 >= model.barrier_price:
                return 0
            res = []
            for path in paths:
                if path[np.where(path >= model.barrier_price)].size != 0:
                    res.append(0)
                else:
                    if model.option_type == "call":
                        res.append(max(path[-1] - model.K, 0))
                    elif model.option_type == "put":
                        res.append(max(model.K - path[-1], 0))
            if res:
                price = np.mean(res)
        elif model.barrier_type == "down":
            if model.S0 <= model.barrier_price:
                return 0
            res = []
            for path in paths:
                if path[np.where(path <= model.barrier_price)].size != 0:
                    res.append(0)
                else:
                    if model.option_type == "call":
                        res.append(max(path[-1] - model.K, 0))
                    elif model.option_type == "put":
                        res.append(max(model.K - path[-1], 0))
            if res:
                price = np.mean(res)
    return np.exp(-model.r * model.T) * price


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
    res = []
    for path in paths:
        temp = path
        if model.lookback_type == 'floating':
            if model.option_type == "call":
                res.append(temp[-1] - min(temp))
            if model.option_type == 'put':
                res.append(max(temp) - temp[-1])
        if model.lookback_type == 'fixed':
            if model.option_type == "put":
                res.append(max(temp) - temp[-1])
            if model.option_type == 'call':
                res.append(max(temp) - temp[-1])
    return np.exp(-model.r * model.T) * np.mean(res)
    