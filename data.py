from config import Config as conf
import os
import numpy as np


def load(path):
    print(path)
    for i in os.listdir(path):
        all = np.load(path + '/' + i)
        img, cond = all[:, :conf.img_size], all[:, conf.img_size:]
        yield (img, cond, i)


def load_data(path=conf.data_path):
    data = dict()
    data["train"] = lambda: load(path + "/train")
    data["test"] = lambda: load(path + "/test")
    data["eval"] = lambda: load(path + "/eval")
    return data
