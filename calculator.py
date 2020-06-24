<<<<<<< HEAD
import numpy as np

call_option = "call"
put_option = "put"
"""
Barrier option
"""
knock_in = "in"
knock_out = "out"
barrier_up = "up"
barrier_down = "down"


def EU_Monte_Carlo(model, paths):
    """
    @param model:
    @param paths:
    @return:
    """
    price = 0
    if model.option_type == call_option:
        price = paths[:, -1] - model.K
    elif model.option_type == put_option:
        price = model.K - paths[:, -1]
    else:
        pass
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
    if model.option_type == call_option:
        tmp = paths[:, -1] - model.K
        price[:, -1] = np.where(tmp > 0, tmp, 0)
    elif model.option_type == put_option:
        tmp = model.K - paths[:, -1]
        price[:, -1] = np.where(tmp > 0, tmp, 0)
    for idx in range(2, step_num + 1):
        back_val = np.exp(-model.r * delta_t) * price[:, -(idx - 1)]
        c_val = np.zeros_like(back_val)
        if model.option_type == call_option:
            c_val = paths[:, -idx] - model.K
        elif model.option_type == put_option:
            c_val = model.K - paths[:, -idx]
        price[:, -idx] = np.where(back_val > c_val, back_val, c_val)
    return np.mean(price[:, 0])


class barrier_Monte_Carlo:
    def __init__(self, knock_type, barrier_type, barrier_price):
        self.knock_type = knock_type
        self.barrier_type = barrier_type
        self.C = barrier_price

    def get(self, model, paths):
        """
        :return: The value of specified barrier option
        """
        price = 0
        if self.knock_type == knock_in:
            if self.barrier_type == barrier_up:
                price = self.up_in(model, paths)
            elif self.barrier_type == barrier_down:
                price = self.down_in(model, paths)
        elif self.knock_type == knock_out:
            if self.barrier_type == barrier_up:
                price = self.up_out(model, paths)
            elif self.barrier_type == barrier_down:
                price = self.down_out(model, paths)
        return np.exp(-model.r * model.T) * price

    def up_out(self, model, paths):
        """
        If the stock price rise across the boundary, the option becomes invalid.
        @param paths:
        @return:
        """
        if model.S0 >= self.C:
            return 0
        res = []
        for path in paths:
            if path[np.where(path >= self.C)].size != 0:
                res.append(0)
            else:
                if model.option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif model.option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)

    def up_in(self, model, paths):
        """
        If the stock price rises across the boundary, the option becomes valid.
        @param paths:
        @return:
        """
        res = []
        for path in paths:
            if path[np.where(path >= self.C)].size == 0:
                res.append(0)
            else:
                if model.option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif model.option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)

    def down_out(self, model, paths):
        """
        If the stock price decline across the boundary, the option becomes invalid.
        @param paths:
        @return:
        """
        if model.S0 <= self.C:
            return 0
        res = []
        for path in paths:
            if path[np.where(path <= self.C)].size != 0:
                res.append(0)
            else:
                if model.option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif model.option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)

    def down_in(self, model, paths):
        """
        If the stock price decline across the boundary, the option becomes valid.
        @param paths:
        @return:
        """
        res = []
        for path in paths:
            if path[np.where(path <= self.C)].size == 0:
                res.append(0)
            else:
                if model.option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif model.option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)
=======
import numpy as np

call_option = "call"
put_option = "put"
"""
Barrier option
"""
knock_in = "in"
knock_out = "out"
barrier_up = "up"
barrier_down = "down"

def EU_Monte_Carlo(model, paths, option_type=call_option):
    """
    @param model:
    @param paths:
    @param option_type:
    @return:
    """
    price = 0
    if option_type == call_option:
        price = paths[:, -1] - model.K
    elif option_type == put_option:
        price = model.K - paths[:, -1]
    else:
        pass
    price = np.where(price > 0, price, 0)
    return np.exp(-model.r * model.T) * np.mean(price)


def AM_Monte_Carlo(model, paths, path_num, step_num, option_type=call_option):
    """
    @param model: Model containing the option parameters
    @param option_type: "call" or "put"
    @param path_num:
    @param step_num:
    @return:
    """
    price = np.zeros_like(paths)
    delta_t = model.T / step_num
    if option_type == call_option:
        tmp = paths[:, -1] - model.K
        price[:, -1] = np.where(tmp > 0, tmp, 0)
    elif option_type == put_option:
        tmp = model.K - paths[:, -1]
        price[:, -1] = np.where(tmp > 0, tmp, 0)
    for idx in range(2, step_num + 1):
        back_val = np.exp(-model.r * delta_t) * price[:, -(idx - 1)]
        c_val = np.zeros_like(back_val)
        if option_type == call_option:
            c_val = paths[:, -idx] - model.K
        elif option_type == put_option:
            c_val = model.K - paths[:, -idx]
        price[:, -idx] = np.where(back_val > c_val, back_val, c_val)
    return np.mean(price[:, 0])


class barrier_Monte_Carlo:
    def __init__(self, knock_type, barrier_type, barrier_price):
        self.knock_type = knock_type
        self.barrier_type = barrier_type
        self.C = barrier_price

    def get(self, model, paths, option_type=call_option, path_num=1000, step_num=1000):
        """
        :return: The value of specified barrier option
        """
        price = 0
        if self.knock_type == knock_in:
            if self.barrier_type == barrier_up:
                price = self.up_in(model, option_type, paths)
            elif self.barrier_type == barrier_down:
                price = self.down_in(model, option_type, paths)
        elif self.knock_type == knock_out:
            if self.barrier_type == barrier_up:
                price = self.up_out(model, option_type, paths)
            elif self.barrier_type == barrier_down:
                price = self.down_out(model, option_type, paths)
        return np.exp(-model.r * model.T) * price

    def up_out(self, model, option_type, paths):
        """
        If the stock price rise across the boundary, the option becomes invalid.
        @param option_type:
        @param paths:
        @return:
        """
        if model.S0 >= self.C:
            return 0
        res = []
        for path in paths:
            if path[np.where(path >= self.C)].size != 0:
                res.append(0)
            else:
                if option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)

    def up_in(self, model, option_type, paths):
        """
        If the stock price rises across the boundary, the option becomes valid.
        @param option_type:
        @param paths:
        @return:
        """
        res = []
        for path in paths:
            if path[np.where(path >= self.C)].size == 0:
                res.append(0)
            else:
                if option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)

    def down_out(self, model, option_type, paths):
        """
        If the stock price decline across the boundary, the option becomes invalid.
        @param option_type:
        @param paths:
        @return:
        """
        if model.S0 <= self.C:
            return 0
        res = []
        for path in paths:
            if path[np.where(path <= self.C)].size != 0:
                res.append(0)
            else:
                if option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)

    def down_in(self, model, option_type, paths):
        """
        If the stock price decline across the boundary, the option becomes valid.
        @param option_type:
        @param paths:
        @return:
        """
        res = []
        for path in paths:
            if path[np.where(path <= self.C)].size == 0:
                res.append(0)
            else:
                if option_type == call_option:
                    res.append(max(path[-1] - model.K, 0))
                elif option_type == put_option:
                    res.append(max(model.K - path[-1], 0))
                else:
                    return 0
        return np.mean(res)
>>>>>>> fc4deffd81807122d92a8903c1d57f4bd1c3da3c
