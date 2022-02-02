import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path




def save_fig(fig, path, name):
    fig.tight_layout()
    path = path + '\\results\\plots\\param_plots'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name.replace('/', ' dev ') + '.jpeg'
    print(path)
    # plt.show()
    fig.savefig(path)
    plt.close(fig)


def normalizedata(data, error):
    normalized_data =  (data - np.min(data)) / (np.max(data) - np.min(data))
    noralized_error = (1/np.max(data))*error
    # print(normalized_data, noralized_error)
    return normalized_data, noralized_error


def transform_table(path, df_mean, df_stabw, properties):

    # use this for sorting samples

    # df_mean['order'] = [3, 0, 1, 4, 5, 2]
    # df_stabw['order'] = [3, 0, 1, 4, 5, 2]
    # df_mean.sort_values(by=['order'], inplace=True)
    # df_stabw.sort_values(by=['order'], inplace=True)
    # df_mean.drop(['order'], axis=1, inplace=True)
    # df_stabw.drop(['order'], axis=1, inplace=True)

    params = df_mean.T.index.tolist()
    for param in params:
        # testen ob Einheit vorliegt
        try:
            unit = param.split()[1]
        except:
            unit = '[]'
        mean = df_mean[param]
        stabw = df_stabw[param]
        df_plot = pd.DataFrame({'mean': mean,'stabw': stabw})
        plot_mean(path, df_plot, param, unit, properties)

def plot_mean(root_path, df_plot, param, unit, properties):
    colors = properties['colors']
    # Create lists for the plot
    samples = df_plot.index.tolist()
    x_pos = np.arange(len(samples))
    mean = df_plot['mean']
    error = df_plot['stabw']
    # mean, error = normalizedata(mean, error)

    fig, ax = plt.subplots()
    barlist = ax.bar(x_pos, mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    for sample, i in zip(samples, range(len(samples))):
        if sample == ' TNT':
            sample = 'TNT'
        barlist[i].set_color(colors[sample])
    ytitle = 'mean ' + unit
    ax.set_ylabel(ytitle)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, root_path, param)

def plot_features(path, properties):
    path_mean = path + '\\results\\mean.csv'
    path_stabw = path + '\\results\\std.csv'
    df_mean = pd.read_csv(path_mean, decimal='.', sep=';')
    df_stabw = pd.read_csv(path_stabw, decimal='.', sep=';')
    df_mean.rename(columns={"Unnamed: 0": "sample"}, inplace=True)
    df_stabw.rename(columns={"Unnamed: 0": "sample"}, inplace=True)
    df_mean.set_index('sample', inplace=True)
    df_stabw.set_index('sample', inplace=True)
    transform_table(path, df_mean, df_stabw, properties)


if __name__ == '__main__':
    path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    plot_param(path)

