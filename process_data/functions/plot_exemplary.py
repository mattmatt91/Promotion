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
from os.path import isfile
import matplotlib



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


def plot_exemplary(df, path, sensor, properties):
    """
    This function plots measurements from the passed DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with data of measurents from one sensor
        path (string): path to root folder
        sensor (string): name of the sensor
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    sensors = properties['sensors']
    plot_properties = properties['plot_properties']['exemplary_plot']
    colors = properties['colors_samples']

    ### for old data ###
    df.columns = [x.replace(' ','').capitalize() for x in df.columns]
    ####################

    fig, ax = plt.subplots(figsize=plot_properties['size'])
    plt.xlim(sensors[sensor]['x_lim_plot'])
    matplotlib.rcParams['legend.fontsize'] = plot_properties['legend_size']
    for i in df.columns:
        ax.plot(df.index, df[i], color=colors[i], label=i, linewidth=1) 
    plt.xlabel('time [s]', fontsize = plot_properties['label_size'])
    plt.ylabel('voltage [V]', fontsize = plot_properties['label_size'])
    plt.legend(loc=0)
    plt.yticks(fontsize=plot_properties['font_size'])
    plt.xticks(fontsize=plot_properties['font_size'])
    plt.tight_layout()
    ax.grid()
    # plt.show()
    save_fig(fig, path, sensor)
    plt.close()

    
def read(path, sensor, root_path, properties):
    """
    This function reads files with the exemplary measurements,
     prepares them and calls the plot function.

    Args:
        path (string): path to file
        sensor (string): name of the sensor
        root_path (string): path to root folder
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    # preparing data frame
    df = pd.read_csv(path, decimal=',', sep='\t')
    df.set_index('time [s]', inplace=True)
    print(sensor)
    plot_exemplary(df, root_path, sensor, properties)


def main(root_path):
    """
    This is the main function of the module.
    It reads the data of the exemplary measurements and plots them.

    Args:
        root_path (string): path to root folder
    """
    properties = extract_properties()
    # finding files
    path_list =[]
    sensors = properties['sensors']
    [path_list.append(root_path + '\\results\\exemplary\\' + x + '_Vergleich.csv') for x in sensors]
    # reading data
    for path, sensor in zip(path_list, sensors):
        if isfile(path):
            print('reading: {0}'.format(path))
        else:
            print('no file found for {0}'.format(sensor))
            continue
        read(path, sensor, root_path, properties)

if __name__ == '__main__':
    root_path = "C:\\Users\\mmuhr-adm\\Desktop\\Test_data"
    main(root_path)