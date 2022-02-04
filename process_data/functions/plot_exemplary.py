"""
This module plots exemplary measurements. A plot is created for each sensor. For these, a file must be created manually for each sensor in which the corresponding samples (column name must correspond to the name of the sample) are entered. In addition, a time axis should be available.
The file name must be created as follows:
**sensorname** _compare.csv
These measurements must be stored in a folder called results//exemplary. 

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from read_files import extract_properties


def save_fig(fig, path, name):
    """
    This function saves the fig object in the folder "results\\plots\\exemplary".

    Args:
        fig (Object): figure to save
        path (string): path to root folder
        name (string): figures name
    """
    fig.tight_layout()
    path = path + 'results\\plots\\exemplary'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    fig.savefig(path)
    plt.close(fig)


def plot_exemplary(df, path, sensor, x_lim_plot, colors):
    """
    This function plots measurements from the passed DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with data of measurents from one sensor
        path (string): path to root folder
        sensor (string): name of the sensor
        x_lim_plot (list): list with lower and upper x limit in seconds
        colors (dictionary): dictionary with colors for sensors
    """
    fig, ax = plt.subplots()
    for i, in df.columns:
        ax.plot(df.index, df[i], color=colors[i], label=i, linewidth=1)
    plt.xlim(x_lim_plot[i][0], x_lim_plot[i][1])
    ax.set(xlabel='time [s]', ylabel='voltage [V]')
    plt.legend()
    ax.grid()
    save_fig(fig, path, sensor)
    plt.show()


def read(path, sensor, root_path, x_lim_plot, colors):
    """
    This function reads files with the exemplary measurements,
     prepares them and calls the plot function.

    Args:
        path (string): path to file
        sensor (string): name of the sensor
        root_path (string): path to root folder
        x_lim_plot (list): list with lower and upper x limit in seconds
        colors (dictionary): dictionary with colors for sensors
    """
    print(path)
    df = pd.read_csv(path, decimal=',', sep='\t')
    df.set_index('time [s]', inplace=True)
    print(sensor)
    plot_exemplary(df, root_path, sensor, x_lim_plot, colors)


def main(root_path):
    """
    This is the main function of the module.
    It reads the data of the exemplary measurements and plots them.

    Args:
        root_path (string): path to root folder
    """
    properties = extract_properties()
    sensors = properties['sensors']
    path_list =[]
    [path_list.append(root_path + '\\results\\exemplary\\' + x + '_Vergleich.csv') for x in sensors]
    for path in path_list:
        print('reading: ', path)
    x_lim_plot = properties['x_lim_plot']
    colors = properties['colors']
    for path, sensor in zip(path_list, sensors):
        read(path, sensor, root_path, x_lim_plot, colors)


if __name__ == '__main__':
    root_path = 'root_path = "C:\\Users\\Matthias\\Desktop\\Messaufbau\\dataaquisition\\data\\test_small'
    main(root_path)