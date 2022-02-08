"""
This module creates plots with all measurements per sample and sensor.

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import plotly.express as px
from os import listdir
from os.path import isfile, join


def save_fig(fig, path, name):
    """
    This function saves the fig object in the folder "results\\plots\\plots_compare".

    Args:
        fig (Object): figure to save
        path (string): path to root folder
        name (string): figures name
    """
    fig.tight_layout()
    # print(path)
    path = path + '\\plots\\plots_compare'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    # print(path)
    # plt.show()
    fig.savefig(path)
    plt.close(fig)

def plot(df, sensor, path, names, properties): #creates plots for every sensor with all measurments   
    """
    This function creates plots from the passed data. One plot per sensor and sample with all measurements.

    Args:
        df (pandas.DataFrame): Dataframe with prepared data from measurements
        name (string): name of sensor and sample
        path (string): path to root folder
        names (): list with name of measurements
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    plot_properties = properties['plot_properties']['compare_plots']
    print('plotting {0}-data'.format(sensor))
    x_lim_plot = properties['sensors'][sensor]['x_lim_plot']
    x_lim_plot_start = x_lim_plot[0]
    x_lim_plot_end = x_lim_plot[1]

    for sample in df.columns.unique():
        title = sensor + '_' + sample
        fig, ax = plt.subplots(figsize=plot_properties['size'], dpi=plot_properties['dpi'])

        #use this for centering around peak
        # t_max = df[sample].max()
        # x_lim_plot_start = t_max - properties['x_lim_plot'][name][0]
        # x_lim_plot_end = t_max + properties['x_lim_plot'][name][1]

        ax.plot(df.index, df[sample])
        plt.xlim(x_lim_plot_start, x_lim_plot_end)
        plt.xlabel(df.index.name, fontsize=plot_properties['label_size'])
        plt.ylabel('voltage [V]', fontsize=plot_properties['label_size'])
        plt.yticks(fontsize=plot_properties['font_size'])
        plt.xticks(fontsize=plot_properties['font_size'])
        ax.grid()
        # plt.show()
        save_fig(fig, path, title)
        plt.close()


def read(path, sensor, root_path, properties):
    """
    This function reads files with the data of the individual sensors
    with all measurements, prepares them and passes them to the plot function.

    Args:
        path (string): path to file
        name (string): name of sensor and sample
        root_path (string): path to foot folder
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    df = pd.read_csv(path, decimal=',', sep=';')
    df.set_index('time [s]', inplace=True)
    names = df.columns.tolist()
    df.columns = [x[:x.find('_')] for x in names]
    plot(df, sensor, root_path, names, properties)



def compare(root_path, properties):
    """
    This is the main function of the module. It reads all files with data
    from one sensor and all measurements. Plots are created.

    Args:
        root_path (string): path to foot folder
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    root_path = root_path + '\\results'
    onlyfiles = [f for f in listdir(root_path) if isfile(join(root_path, f))]
    files_list = [root_path + '\\' + i for i in onlyfiles if i.find('gesamt') >= 0]
    sensors = [i[:i.rfind('_')] for i in onlyfiles if i.find('gesamt') >= 0]
    for path, sensor in zip(files_list, sensors):
        read(path, sensor, root_path, properties)


if __name__ == '__main__':
    root_path = 'E:\\Promotion\\Daten\\29.06.21_Paper_reduziert'
    compare(root_path)


    