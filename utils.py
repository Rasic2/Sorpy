import numpy as np
from collections import defaultdict

"""
Utils functions

- format_dict:  格式化字典输出
- distance:     计算欧氏距离
"""

def format_dict(dict_i):
    strings = ""
    for key, value in dict_i.items():
        strings += f"{key:20s}:  {value} \n"

    return strings.rstrip()

def format_defaultdist(dist_i):
    for key, value in dist_i.items():
        yield (key, value[0])

def distance(array_i, array_j):
    return np.linalg.norm(array_i-array_j)


class Format_defaultdict(defaultdict):
    def __repr__(self):
        strings=""
        for key, value in self.items():
            strings += f"{key} <---> {value[0]} \n"
        return strings

class Format_list(list):
    def __repr__(self):
        strings=""
        for item in self:
            strings += f"{item} \n"
        return strings.rstrip()
