import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
from functools import wraps
import seaborn as sns
from common.logger import root_dir, logger


def df_create(dname):
    results = {}
    for ii in range(50):
        fpath = dname / f"summary/{ii + 1}_iter"
        gpath = dname / f"summary/{ii + 1}_force"
        temp_dict = defaultdict(list)
        with open(fpath, "r") as f, open(gpath, "r") as g:
            cfg_iter = f.readlines()
            cfg_force = g.readlines()

        i = 0
        length = len(cfg_iter) / 2
        while i < length:
            temp_dict[i + 1].append(int(cfg_iter[2 * i].split()[1]))
            temp_dict[i + 1].append(float(cfg_iter[2 * i + 1].split()[2]))
            temp_dict[i + 1].append(float(cfg_force[i].split()[5]))
            i += 1
        df = pd.DataFrame(pd.DataFrame(temp_dict).values.T, index=[i + 1 for i in range(int(length))],
                          columns=['iter', 'energy', 'force'])
        df['iter'] = df['iter'].astype(int)
        results[f'file_{ii + 1}'] = df
    return results


def plot_wrap(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        figure = plt.figure(figsize=(7, 5))
        plt.rc('font', family='Arial')  # <'Times New Roman'>
        plt.rcParams['mathtext.default'] = 'regular'  # 配置数学公式字体
        func(*args, **kargs)

    return wrapper


@plot_wrap
def plot_steps(ori_dfs, ML_dfs):
    ori_isteps, ml_isteps = [], []
    for key, df in ori_dfs.items():
        ori_isteps.append(len(df.index))

    for key, df in ML_dfs.items():
        ml_isteps.append(len(df.index))
    ori_isteps, ml_isteps = np.array(ori_isteps), np.array(ml_isteps)

    b_min, b_diff = [], []
    count = 0
    for i, j in zip(ori_isteps, ml_isteps):
        count += 1
        diff = i - j
        if diff > 0:
            b_min.append(j)
            b_diff.append(diff)
            plt.bar([count], b_min[-1], width=0.35, color='#FF8025')
            plt.bar([count], b_diff[-1], width=0.35, bottom=b_min[-1], color='#1176B2')
        else:
            b_min.append(i)
            b_diff.append(-diff)
            plt.bar([count], b_min[-1], width=0.35, color='#1176B2')
            plt.bar([count], b_diff[-1], width=0.35, bottom=b_min[-1], color='#FF8025')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Steps", fontsize=22)
    # plt.show()
    plt.savefig(f"{root_dir}/results/analysis/istep.svg")


@plot_wrap
def plot_maxforce(*results, save=True):
    for result in results:
        maxf = []
        for key, df in result.items():
            maxf.append(df['force'].max())
        plt.plot([i + 1 for i in range(len(maxf))], maxf, '-o')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Max Force", fontsize=22)

    if save:
        plt.savefig(f"{root_dir}/results/analysis/mforce.svg")
    else:
        plt.show()


@plot_wrap
def plot_maxiter(*results, save=True):
    for result in results:
        maxiter = []
        for key, df in result.items():
            maxiter.append(df['iter'].mean())
        plt.plot([i + 1 for i in range(len(maxiter))], maxiter, '-o')

    ax = plt.gca()
    y_major_locator = plt.MultipleLocator(2)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Mean Iter nums", fontsize=22)

    if save:
        plt.savefig(f"{root_dir}/results/analysis/mean_iter.svg")
    else:
        plt.show()


@plot_wrap
def plot_energy(*results, save=True):
    for result in results:
        energy = []
        for key, df in result.items():
            energy.append(df['energy'].values[-1])
        plt.plot([i + 1 for i in range(len(energy))], energy, '-o')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Energy", fontsize=22)

    if save:
        plt.savefig(f"{root_dir}/results/analysis/energy.svg")
    else:
        plt.show()


def plot_time():
    with open("ori_summary/cpu_spent") as f, open("ML_summary/cpu_spent") as g:
        cfg_ori = f.readlines()
        cfg_ml = g.readlines()
    ori_time = [float(item.split()[6]) for item in cfg_ori]
    ml_time = [float(item.split()[6]) for item in cfg_ml]
    ori_time, ml_time = np.array(ori_time) / 3600, np.array(ml_time) / 3600

    b_min, b_diff = [], []
    count = 0
    for i, j in zip(ori_time, ml_time):
        count += 1
        diff = i - j
        if diff > 0:
            b_min.append(j)
            b_diff.append(diff)
            plt.bar([count], b_min[-1], width=0.35, color='#FF8025')
            plt.bar([count], b_diff[-1], width=0.35, bottom=b_min[-1], color='#1176B2')
        else:
            b_min.append(i)
            b_diff.append(-diff)
            plt.bar([count], b_min[-1], width=0.35, color='#1176B2')
            plt.bar([count], b_diff[-1], width=0.35, bottom=b_min[-1], color='#FF8025')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Time Spent / h", fontsize=22)
    # plt.show()
    plt.savefig(f"{root_dir}/results/analysis/time_spent.svg")


@plot_wrap
def plot_violin(*dirs, save=True):
    time_spent = []
    kind = []
    for file in dirs:
        with open(Path(file) / "cpu_spent", "r") as f:
            cfg = [(float(item.split()[6]), file.name) for item in f.readlines()]
            logger.info(f"{Path(file).name}: avg time spent: {np.array([i for i, _ in cfg]).mean() / 3600:.3} h")
            time_spent.append([i for i, _ in cfg])
            kind.append([i for _, i in cfg])

    time_spent = np.array(sum(time_spent, [])) / 3600
    kind = sum(kind, [])

    results = pd.DataFrame({'TimeSpent': time_spent, 'Type': kind})
    sns.violinplot(data=results, x="Type", y="TimeSpent")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("")
    plt.ylabel("Time Spent / h", fontsize=22)
    if save:
        plt.savefig(f"{root_dir}/results/analysis/time_spent_violin.svg")
    else:
        plt.show()


def plot_index(index, *results, save=True):
    for item in ['iter', 'energy', 'force']:
        plt.figure()
        ax = plt.subplot(111)
        for result in results:
            df = result[f'file_{index}']
            ax.plot(df.index, df[item], '-o')

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylabel(item, fontsize=22)
        if save:
            plt.savefig(f"{root_dir}/results/analysis/{index}_{item}.svg")
        else:
            plt.show()


if __name__ == "__main__":
    ori_dir = Path("test_set/summary/ori")
    ML_v2_dir = Path("test_set/summary/ML_v2")
    ML_xdat_m_dir = Path("test_set/summary/ML-xdat-m")
    ML_xdat_o_dir = Path("test_set/summary/ML-xdat-o")
    ML_xdat_m2_dir = Path("test_set/summary/ML-xdat-m-iter-0.005")

    ori_results = df_create(ori_dir)
    ML_v2_results = df_create(ML_v2_dir)
    ML_xdat_m_results = df_create(ML_xdat_m_dir)
    ML_xdat_o_results = df_create(ML_xdat_o_dir)
    ML_xdat_m2_results = df_create(ML_xdat_m2_dir)

    # plot_steps(ori_results, ML_results)
    # plot_maxforce(ori_results, ML_v2_results, ML_xdat_m_results)
    # plot_maxiter(ori_results, ML_v2_results, ML_xdat_m_results)
    # plot_energy(ori_results, ML_v2_results, ML_xdat_m_results, save=False)
    # plot_time()
    plot_violin(ori_dir, ML_v2_dir, ML_xdat_m_dir, ML_xdat_o_dir, ML_xdat_m2_dir, save=False)
    # plot_index(30, ori_results, ML_v2_results, ML_xdat_m_results, ML_xdat_o_results, save=False)
