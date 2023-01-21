from RL_segbot.testConstants import *

import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import dill
import copy
from collections import defaultdict
import seaborn as sns;

sns.set()
sns.set_style("whitegrid")

SPINE_COLOR = 'gray'


def latexify(fig_width=None, fig_height=None, columns=1, labelsize=10):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    See https://nipunbatra.github.io/blog/2014/latexify.html for more detail.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    labelsize : int, optional
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    MAX_HEIGHT_INCHES = 10.0
    LABELSIZE = labelsize

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    # if fig_height > MAX_HEIGHT_INCHES:
    #     print("WARNING: fig_height too large: {0} so will reduce to {1}inches.".format(fig_height, MAX_HEIGHT_INCHES))
    #     fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'pdf',
              # 'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': LABELSIZE * 1.2,  # fontsize for x and y labels (was 10)
              'axes.titlesize': LABELSIZE * 1.2,
              'patch.linewidth': 0.1,
              'patch.edgecolor': 'white',
              'legend.fontsize': LABELSIZE,  # was 10
              'legend.loc': 'upper right',
              'legend.borderpad': 0.1,
              'xtick.labelsize': LABELSIZE,
              'ytick.labelsize': LABELSIZE,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


# Get the counts for the agent to visit each state
def get_count_nodes(_mdp, episode=None):
    count_dict = defaultdict(list)
    for node in _mdp.G:
        count_dict[str(node)].append(_mdp.G.nodes[node]['count'])
    _df = pd.DataFrame.from_dict(count_dict)
    _df["episode"] = episode
    return _df


def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",
                           H=None, **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) - 1
    n_ind = len(dfall[0].index)
    # print(n_df, n_col, n_ind)
    fig, axe = plt.subplots()

    for df in dfall:  # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0.05,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      # colormap="Set3",
                      color=['#d2d2f7', '#f4bead', '#a9fab8', '#f3e051', '#f3c1f8', '#c4fdf6', '#fbc677'],
                      edgecolor="white",
                      **kwargs)  # make bar plots
    # print(axe)
    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    # print(len(h), h)
    # print(len(l), l)
    x = float(n_df + 1)
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + (1 / x * i - 0.5) / float(n_col))
                # rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_hatch(H[int(i / n_col)])  # edited part
                rect.set_width(1 / x)

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / x) / 2.)
    axe.set_xticklabels(df.index, rotation=0)
    axe.set_ylabel("The number of times the agent visited")
    axe.set_xlabel("Episode")
    # axe.set_ylim(0, 15000)
    # axe.set_title(title)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        # n.append(axe.bar(0, 0, color="gray", hatch=H * i))
        n.append(axe.bar(0, 0, color="gray", hatch=H[i]))

    l1 = axe.legend(h[:n_col], l[:n_col], loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)
    if labels is not None:
        l2 = plt.legend(n, labels, loc='lower left', bbox_to_anchor=(1.01, 0.0))
    axe.add_artist(l1)
    return axe


def window(x, win=10):
    tmp = np.array(range(len(x)), dtype=float)
    counter = 0
    while counter < len(x):
        tmp[counter] = float(x[counter:counter + win].mean())
        if len(x[counter:]) < win:
            tmp[counter:] = float(x[counter:].mean())
        counter += 1
    return pd.Series(tmp)


def createFigure(data, x, y, hue, filename, loc='lower left', bbox=(-0.1, 1.02, 1.1, 0.2), mode='expand', ncol=5,
                 palette_size=None):
    latexify(10, labelsize=15)  # TODO: point
    fig, ax = plt.subplots()
    if palette_size is None:
        sns.lineplot(x=x, y=y, hue=hue, data=data, ax=ax)
    else:
        palette = sns.color_palette("Set1", palette_size)
        sns.lineplot(x=x, y=y, hue=hue, palette=palette, data=data, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    plt.legend(handles=handles[1:], title=None, loc=loc, bbox_to_anchor=bbox, mode=mode, ncol=ncol)
    plt.tight_layout()
    format_axes(ax)  # TODO: point
    plt.savefig(filename)
    del fig
    del ax


def compareMethodsRewards(_mdp, _agent, _window=50):
    import glob
    csvfiles = []
    for a in _agent:
        # csvfiles += glob.glob(CSV_DIR + "{0}_{1}_*".format(a, _mdp))
        csvfiles += glob.glob("real_experiment/" + "{0}_*30*_fin*".format(a))
    df = pd.read_csv(csvfiles[0])
    # df["Cumulative Reward"] = df["Cumulative Reward"].rolling(window=_window).mean()
    df["Cumulative Reward"] = df["Cumulative Reward"].rolling(window=5).mean()
    for f in csvfiles[1:]:
        tmp = pd.read_csv(f)
        # tmp["Cumulative Reward"] = tmp["Cumulative Reward"].rolling(window=_window).mean()
        tmp["Cumulative Reward"] = tmp["Cumulative Reward"].rolling(window=5).mean()
        df = df.append(tmp, ignore_index=True)

    data = df.loc[:, ["Episode", "Cumulative Reward", "AgentName", "seed"]]
    print(data)
    # createFigure(data, x="Episode", y="Cumulative Reward", hue="AgentName",
    #              filename=FIG_DIR + "REWARD_{0}.pdf".format(_mdp))
    createFigure(data, x="Episode", y="Cumulative Reward", hue="AgentName",
                 filename="real_experiment/" + "REWARD_realS0G15E30T10.pdf", mode=None, loc="lower center")


def compareVariablesRewards(_mdp, _agent, _window=50, prefix="", xlabel="gamma"):
    import glob
    csvfiles = glob.glob(CSV_DIR + "testenvforparam/{0}_{1}*".format(_agent, _mdp))
    # print("{0}_{1}".format(_agent, _mdp))
    tmp = pd.read_csv(csvfiles[0])
    tmp["Cumulative Reward"] = tmp["Cumulative Reward"].rolling(window=_window).mean()
    df = tmp.tail(1)
    for f in csvfiles[1:]:
        tmp = pd.read_csv(f)
        tmp["Cumulative Reward"] = tmp["Cumulative Reward"].rolling(window=_window).mean()
        df = df.append(tmp.tail(1), ignore_index=True)

    data = df.loc[:, [xlabel, "Cumulative Reward", "MDPName", "seed", "AgentName", "alpha"]]
    # print(type(data['alpha'].map('{:.2f}'.format)))
    data.loc[:, "alpha"] = data["alpha"].map("{:.2f}".format)
    print(data.head())
    createFigure(data, x=xlabel, y="Cumulative Reward", hue="alpha", palette_size=9,
                 filename=FIG_DIR + "VarPerf_{0}_{1}".format(_agent, prefix))


def compareMethodsSteps(_mdp, _agent, _window=50):
    import glob
    csvfiles = []
    for a in _agent:
        csvfiles += glob.glob(CSV_DIR + "{0}_{1}_*".format(a, _mdp))
    df = pd.read_csv(csvfiles[0])
    df["Timestep"] = df["Timestep"].rolling(window=_window).mean()
    for f in csvfiles[1:]:
        tmp = pd.read_csv(f)
        tmp["Timestep"] = tmp["Timestep"].rolling(window=_window).mean()
        df = df.append(tmp, ignore_index=True)

    data = df.loc[:, ["Episode", "Timestep", "AgentName", "seed"]]
    createFigure(data, x="Episode", y="Timestep", hue="AgentName", filename=FIG_DIR + "REWARD_{0}.pdf".format(_mdp))


def _load(_mdp, _agent):
    import glob
    pkls = glob.glob(PKL_DIR + "mdp_{0}_{1}*fin.pkl".format(_agent, _mdp))
    # print(pkls)
    df_dict = defaultdict(pd.DataFrame)
    for i, pkl in enumerate(pkls):
        with open(pkl, "rb") as f:
            _mdp = dill.load(f)

        count_df = get_count_nodes(_mdp)
        # set episode as index
        count_df = count_df.set_index('episode')
        df_dict[i] = count_df
    return sum(df_dict.values()) / len(pkls)


def countNumStateVisited(_mdp, _agents):
    df_dict = defaultdict(pd.DataFrame)
    new_df_dict = copy.deepcopy(df_dict)
    for _agent in _agents:
        df = _load(_mdp, _agent)
        new_df = pd.DataFrame(columns=['Area1', 'Area2', 'Area3', 'Area4', 'Area5', 'Area6'])

        area1 = ['s0', 's1', 's2']
        area2 = ['s3', 's5', 's6']
        area3 = ['s4', 's8', 's9']
        area4 = ['s11', 's12']
        area5 = ['s13', 's14']
        area6 = ['s7', 's10', 's15']
        # print(df.loc[:, area1].sum(axis=1))
        new_df['Area1'] = df.loc[:, area1].sum(axis=1)
        new_df['Area2'] = df.loc[:, area2].sum(axis=1)
        new_df['Area3'] = df.loc[:, area3].sum(axis=1)
        new_df['Area4'] = df.loc[:, area4].sum(axis=1)
        new_df['Area5'] = df.loc[:, area5].sum(axis=1)
        new_df['Area6'] = df.loc[:, area6].sum(axis=1)
        new_df = new_df / new_df.iloc[:, :].sum(axis=1).values[0]

        list_sum_total = [[new_df.iloc[0, 0], new_df.iloc[0, 1], new_df.iloc[0, 2], new_df.iloc[0, 3],
                           new_df.iloc[0, 4], new_df.iloc[0, 5]]]
        total = sum(list_sum_total[0])
        tmp = np.round(np.array([new_df.iloc[0, 0], new_df.iloc[0, 1], new_df.iloc[0, 2], new_df.iloc[0, 3],
                                 new_df.iloc[0, 4], new_df.iloc[0, 5]]) / total * 100, 2)
        list_sum_total.append(tmp)
        save_df = pd.DataFrame(list_sum_total, columns=['Area1', 'Area2', 'Area3', 'Area4', 'Area5', 'Area6'])
        save_df.to_csv(CSV_DIR + "count_{0}_{1}.csv".format(_mdp, _agent))

        df["Name"] = _agent
        new_df["Name"] = _agent
        df_dict[_agent] = df
        new_df_dict[_agent] = new_df

    H = ["", "//", "..", "xx", "\\\\"]
    latexify(10, labelsize=15)  # TODO: point
    # axe = plot_clustered_stacked([df for df in new_df_dict.values()], [name for name in new_df_dict.keys()], H=H)
    axe = plot_clustered_stacked([df for df in new_df_dict.values()], labels=list(new_df_dict.keys()), H=H)
    axe.xaxis.grid(False)
    axe.yaxis.grid(True)
    plt.tight_layout()
    format_axes(axe)  # TODO: point
    plt.savefig(FIG_DIR + "count_{0}.pdf".format(_mdp))


def countNumStateVisitedRatio(_mdp, _agents):
    df_dict = defaultdict(pd.DataFrame)
    new_df_dict = copy.deepcopy(df_dict)
    for _agent in _agents:
        df = _load(_mdp, _agent)
        new_df = pd.DataFrame(columns=['Area1', 'Area2', 'Area3', 'Area4', 'Area5', 'Area6'])

        area1 = ['s0', 's1', 's2']
        area2 = ['s3', 's5', 's6']
        area3 = ['s4', 's8', 's9']
        area4 = ['s11', 's12']
        area5 = ['s13', 's14']
        area6 = ['s7', 's10', 's15']
        # print(df.loc[:, area1].sum(axis=1))
        new_df['Area1'] = df.loc[:, area1].sum(axis=1)
        new_df['Area2'] = df.loc[:, area2].sum(axis=1)
        new_df['Area3'] = df.loc[:, area3].sum(axis=1)
        new_df['Area4'] = df.loc[:, area4].sum(axis=1)
        new_df['Area5'] = df.loc[:, area5].sum(axis=1)
        new_df['Area6'] = df.loc[:, area6].sum(axis=1)
        new_df = new_df / new_df.iloc[:, :].sum(axis=1).values[0]
        # print(new_df)

        df["Name"] = _agent
        new_df["Name"] = _agent
        df_dict[_agent] = df
        new_df_dict[_agent] = new_df

    H = ["", "//", "..", "xx", "\\\\"]
    latexify(10, labelsize=15)  # TODO: point
    # axe = plot_clustered_stacked([df for df in new_df_dict.values()], [name for name in new_df_dict.keys()], H=H)
    axe = plot_clustered_stacked([df for df in new_df_dict.values()], labels=list(new_df_dict.keys()), H=H)
    axe.xaxis.grid(False)
    axe.yaxis.grid(True)
    plt.tight_layout()
    format_axes(axe)  # TODO: point
    plt.savefig(FIG_DIR + "count_{0}_ratio.pdf".format(_mdp))


def parseOptions():
    import optparse
    optParser = optparse.OptionParser()
    optParser.add_option('-w', '--window', action='store',
                         type='int', dest='window', default=50,
                         help='Window size (default %default)')
    optParser.add_option('-a', '--agent', action='store', metavar="A",
                         type='string', dest='agent', default="gdq",
                         help='Agent type (options are \'q-learning\', \'sarsa\', \'rmax\', \'dynaq\', and \'gdq\' default %default)')
    optParser.add_option('--mdp', action='store', metavar="M",
                         type='string', dest='mdp', default="env1S0G17E500I25",
                         help='MDP name (options are \'testGraph\', \'envS0G14\', \'envS0G16\', \'envS10G14\', \'envS10G16\', \'envS21G14\', and \'envS21G16\' default %default)')
    optParser.add_option('-c', '--compare', action='store', metavar="C",
                         type='string', dest='compare', default="methods",
                         help='compare option (options are \'variables\', \'methods\', and \'both\' default %default)')
    optParser.add_option('-x', '--xlabel', action='store', metavar="C",
                         type='string', dest='xlabel', default="gamma",
                         help='xlabel option (options are \'gamma\', \'alpha\', \'epsilon\', \'rmax\', \'ucount\', and \'lookahead\' default %default)')
    optParser.add_option('-p', '--prefix', action='store', metavar="P",
                         type='string', dest='prefix', default="",
                         help='filename prefix')

    opts, args = optParser.parse_args()

    return opts


def process_csv_of_exp():
    file_list = list()
    for i in range(0, 25):
        file_list.append("GDQ_realS0G15E25T25_0_{0}.csv".format(i))

    df = pd.DataFrame()
    for filename in file_list:
        tmp = pd.read_csv("real_experiment/" + filename)
        df = df.append(tmp.loc[:, ["Episode","Timestep","Cumulative Reward","seed","AgentName","MDPName","alpha","gamma","epsilon","rmax","ucount","lookahead"]])
    print(df)
    df.to_csv("real_experiment/GDQ_realS5G15E25T10_0_fin.csv")



if __name__ == "__main__":
    # process_csv_of_exp()
    # exit()


    opts = parseOptions()
    print(opts.mdp)
    if opts.compare == 'methods':
        METHOD = ["DynaQ", "GDQ"]
        # METHOD = ["GDQ"]
        # countNumStateVisitedRatio(opts.mdp, METHOD)
        # countNumStateVisited(opts.mdp, METHOD)
        compareMethodsRewards(opts.mdp, METHOD, _window=opts.window)

    elif opts.compare == 'variables':
        if opts.agent == 'dynaq':
            agent = "DynaQ"
        elif opts.agent == 'gdq':
            agent = "GDQ"
        else:
            agent = "GDQ"
        compareVariablesRewards(opts.mdp, agent, _window=opts.window, prefix=opts.prefix, xlabel=opts.xlabel)

    elif opts.both:
        METHOD = ["DynaQ", "GDQ"]
        print("compare methods amongst {0} in {1}".format(METHOD, opts.mdp))
        compareMethodsRewards(opts.mdp, METHOD, _window=opts.window)

        if opts.agent == 'dynaq':
            agent = "DynaQ"
        elif opts.agent == 'gdq':
            agent = "GDQ"
        else:
            agent = opts.agent
        print("compare variables of {0}".format(agent))
        compareVariablesRewards(agent, _window=opts.window, prefix=opts.prefix)
