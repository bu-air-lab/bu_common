import glob
from collections import defaultdict

import dill
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from RL.testConstants import *

METHOD = ("MBRLP", "QLearning", "Sarsa", "DynaQ", "RMAX")
MDP = ("easy", "normal", "hard", "extreme")


# Import mdp
def import_mdp(file_path):
    with open(file_path, "rb") as f:
        _mdp = dill.load(f)
    return _mdp


###########################################################################################################
# For Counts for the agent to visit each state Figure
###########################################################################################################
# Get the counts for the agent to visit each state
def get_count_nodes(_mdp):
    count_dict = defaultdict(list)
    for node in _mdp.G:
        count_dict[node].append(_mdp.G.nodes[node]['count'])
    _df = pd.DataFrame.from_dict(count_dict)
    return _df


# Get DataFrame of the counts of each run.
def get_df_count_nodes(f_list):
    _mdp = import_mdp(f_list[0])
    tmp_df = get_count_nodes(_mdp)
    for file_path in f_list[1:]:
        _mdp = import_mdp(file_path)
        tmp_df = tmp_df.append(get_count_nodes(_mdp))
    return tmp_df


def barplot(df, filename):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
    sns.barplot(x="x", y="y", hue="method", data=df.where(df["x"].isin(df["x"][:6])), ax=axs[0][0])
    sns.barplot(x="x", y="y", hue="method", data=df.where(df["x"].isin(df["x"][6:12])), ax=axs[0][1])
    sns.barplot(x="x", y="y", hue="method", data=df.where(df["x"].isin(df["x"][12:19])), ax=axs[1][0])
    sns.barplot(x="x", y="y", hue="method", data=df.where(df["x"].isin(df["x"][-7:])), ax=axs[1][1])
    plt.show()
    plt.savefig("figures/" + filename[4:-4] + ".png")
    plt.close()


def make_fig_counter(pkl, csv):
    for mdp in MDP:
        df = pd.DataFrame(columns={"x", "y", "method"})
        for method in METHOD:
            tmp = pd.DataFrame(columns={"x", "y", "method"})
            # print(method, mdp)
            file_list = glob.glob(pkl + "mdp_{0}_{1}_*.pkl".format(method, mdp))
            count_df = get_df_count_nodes(file_list)
            # print(count_df)
            # print(count_df.mean())
            count_df.mean().to_csv(csv + "count_{0}_{1}.csv".format(method, mdp))

            tmp["x"] = list(map(str, count_df))
            tmp["y"] = list(count_df.mean())
            tmp["method"] = [method] * len(tmp["x"])
            df = df.append(tmp)

        # print(df)
        barplot(df, csv + "count_{0}.csv".format(mdp))


###########################################################################################################
# For Success Rate Figure
###########################################################################################################
# Import csv
def get_csv(file_path, method, success_epi):
    tmp = pd.read_csv(file_path, index_col=0)
    tmp_df = pd.DataFrame(columns={'episode', 'success_rate', 'seed', 'method'})
    df = pd.DataFrame(columns={'episode', 'success_rate', 'seed', 'method'})
    for col in tmp.columns:
        tmp_df['episode'] = tmp.index.values.tolist()
        tmp_df['success_rate'] = tmp.loc[:, col] < success_epi
        tmp_df['seed'] = [col] * len(list(tmp.index.values))
        tmp_df['method'] = [method] * len(list(tmp.index.values))
        df = df.append(tmp_df)
    return df


# Concatenate data frames for a process
def concat_df(f_list, method, success_epi):
    df = get_csv(f_list[0], method, success_epi)
    for file_path in f_list[1:]:
        tmp_df = get_csv(file_path, method, success_epi)
        df = df.append(tmp_df)
    return df


def make_csv_su(directory, filename, success_epi):
    df = pd.DataFrame(columns={'episode', 'success_rate', 'seed', 'method'})
    for method in METHOD:
        # print(method, mdp)
        file_list = glob.glob(directory + "timestep_{0}_{1}_9.csv".format(method, mdp))
        print(file_list)
        tmp = concat_df(file_list, method, success_epi)

        # tmp["x"] = list(map(str, success_rate_df))
        # tmp["y"] = list(success_rate_df.mean())
        # tmp["method"] = [method] * len(tmp["x"])
        df = df.append(tmp)

    print(df)
    df.to_csv(directory + filename, index=False)
    # barplot(df, CSV_DIR + "count_{0}.csv".format(mdp))


def mean_csv(filename, resize=50):
    load_df = pd.read_csv(filename)
    df = pd.DataFrame(columns={'episode', 'success_rate', 'seed', 'method'})
    for method in METHOD:
        sub_load_df = load_df.where(load_df['method'] == method).dropna()
        # print(sub_load_df)
        tmp = pd.DataFrame(columns={'episode', 'success_rate', 'seed', 'method'})
        seed_num = max(list(map(int, load_df['seed'].values))) + 1
        episode_num = max(list(map(int, sub_load_df['episode'].values))) + 1
        resized_sub_index = np.resize(sub_load_df.index, (seed_num, episode_num))
        # print(resized_sub_index)
        for s, one_episode in enumerate(resized_sub_index):
            window_list = list()
            s_tmp = pd.DataFrame(columns={'episode', 'success_rate', 'seed', 'method'})
            # print(one_episode)
            resized_one_episode = np.resize(one_episode, (int(len(one_episode) / resize), resize))
            for step in resized_one_episode:
                window_list.append(sub_load_df.loc[step, 'success_rate'].mean())
            # print(len(one_episode), window_list)
            s_tmp['success_rate'] = window_list
            s_tmp['episode'] = range(resize, len(one_episode) + 1, resize)
            s_tmp['seed'] = [s] * len(window_list)
            s_tmp['method'] = [method] * len(window_list)
            tmp = tmp.append(s_tmp)
        df = df.append(tmp)
    df.to_csv(filename, index=False)


def make_fig_su_rate(filename, loc='lower right', pos=(1, 0)):
    tmp_df = pd.read_csv(filename)
    # print(tmp_df)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()
    sns.lineplot(x='episode', y='success_rate', hue='method', data=tmp_df, style="method", err_style="bars", ax=ax)
    # plt.title(filename)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], loc=4)
    plt.legend(loc=loc, bbox_to_anchor=pos, ncol=1)
    plt.savefig(FIG_DIR + filename[4:-4] + ".png")
    del fig
    del ax


if __name__ == "__main__":
    step = 50
    episode = 2000
    window = 30
    SUFFIX = "{0}step{1}epi/".format(step, episode)
    DIRNAME = CSV_DIR + SUFFIX

    # make_fig_counter(PKL_DIR + SUFFIX, CSV_DIR + SUFFIX)
    for mdp in MDP:
        name = "success_rate_{0}win{1}step{2}epi_{3}.csv".format(window, step, episode, mdp)
        path = DIRNAME + name
        make_csv_su(CSV_DIR + SUFFIX, name, step)
        print(path)
        mean_csv(path, window)
        make_fig_su_rate(path)
