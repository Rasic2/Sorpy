import profile
import pstats
from collections import defaultdict
from functools import wraps

import numpy as np
from matplotlib import pyplot as plt

"""
- <class format_dict>           格式化字典输出
- <class format_defaultdist>    格式化列表字典输出
- <func distance>               计算欧氏距离
- <func normalize_coord>        坐标归一化处理
- <func plot_class_wrap>        绘图类装饰器
"""


def format_dict(dict_i):
    strings = ""
    for key, value in dict_i.items():
        strings += f"{key:20s}:  {value} \n"

    return strings.rstrip()


def format_defaultdist(dist_i):
    for key, value in dist_i.items():
        yield key, value[0]


def distance(array_i, array_j):
    return np.linalg.norm(array_i - array_j)


def normalize_coord(data):
    data[:, 37, 2] = np.where(data[:, 37, 2] >= 0, data[:, 37, 2], 360 + data[:, 37, 2])
    data[:, 37, :] = data[:, 37, :] / [1, 180, 360] - [1.142, 0, 0]
    return data


def plot_clsss_wrap(func):
    @wraps(func)
    def wrapper(self, *args, **kargs):
        self.figure = plt.figure(figsize=(9, 6))
        plt.rc('font', family='Arial')  # <'Times New Roman'>
        plt.rcParams['mathtext.default'] = 'regular'
        func(self, *args, **kargs)

    return wrapper


def performance(func: str = "main()", fname: str = "TimeCost"):
    profile.run(func, fname)
    p = pstats.Stats(fname)
    p.strip_dirs().sort_stats("time").print_stats()


# class Format_defaultdict(defaultdict):
#     def __repr__(self):
#         strings = ""
#         for key, value in self.items():
#             strings += f"{key} <---> {value[0]} \n"
#         return strings


class Format_list(list):
    def __repr__(self):
        strings = ""
        for item in self:
            strings += f"{item} \n"
        return strings.rstrip()
