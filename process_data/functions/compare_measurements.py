import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import plotly.express as px


def save_fig(fig, path, name):
    fig.tight_layout()
    # print(path)
    path = path + '\\plots\\plots'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    # print(path)
    # plt.show()
    fig.savefig(path)
    plt.close(fig)

def plot(df, name, path, names, properties): #creates plots for every sensor with all measurments
    print('plotting {0}-data'.format(name))
    x_lim_plot = properties['x_lim_plot']
    x_lim_plot_start = x_lim_plot[name][0]
    x_lim_plot_end = x_lim_plot[name][1]

    for sample, sample_name in zip(set(df.columns.tolist()), names):
        title = name + '_' + sample
        fig, ax = plt.subplots()

        #use this for centering around peak
        # t_max = df[sample].max()
        # x_lim_plot_start = t_max - properties['x_lim_plot'][name][0]
        # x_lim_plot_end = t_max + properties['x_lim_plot'][name][1]

        
        ax.plot(df.index, df[sample])
        plt.xlim(x_lim_plot_start, x_lim_plot_end)
        ax.set(xlabel=df.index.name, ylabel='voltage [V]')
        # plt.show()
        ax.grid()
        save_fig(fig, path, title)

def read(path, name, root_path, properties):
    df = pd.read_csv(path, decimal=',', sep=';')
    df.set_index('time [s]', inplace=True)
    names = df.columns
    df.columns = [x[:x.find('_')] for x in df.columns.tolist()]
    plot(df, name, root_path, names, properties)



def compare(root_path, properties):
    root_path = root_path + '\\results'
    sensors = properties['sensors']
    path_list =[]
    [path_list.append(root_path + '\\' + x + '_gesamt.csv') for x in sensors ]
    for path, sensor in zip(path_list, sensors):
        read(path, sensor, root_path, properties)


if __name__ == '__main__':
    root_path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    compare(root_path)