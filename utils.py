"""
Utils functions

- format_dict: 格式化字典输出
"""

def format_dict(dict_i):
    strings = ""
    for key, value in dict_i.items():
        strings += f"{key:20s}:  {value} \n"

    return strings.rstrip()