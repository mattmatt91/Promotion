"""
This module pltots  all extraceted features with mean and standard dev.

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_fig(fig, path, name):
    """
    This function saves the fig object in the folder "results\\plots\\param_plots".

    Args:
        fig (Object): figure to save
        path (string): path to root folder
        name (string): figures name
    """
    fig.tight_layout()
    path = path + '\\results\\plots\\param_plots'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name.replace('/', ' dev ') + '.jpeg'
    print(path)
    # plt.show()
    fig.savefig(path)
    plt.close(fig)


def normalizedata(data, error):
    """
    This function normalises data, including errors, between 1 and 0.

    Args:
        data (list): list with data to normalise
        error (list): list with to data corresponding errors

    Returns:
        normalised_data (list): normalised data
        normalised_error (list): normalised error
    """
    normalized_data =  (data - np.min(data)) / (np.max(data) - np.min(data))
    noralized_error = (1/np.max(data))*error
    # print(normalized_data, noralized_error)
    return normalized_data, noralized_error


def transform_table(path, df_mean, df_stabw, properties):
    """
    This function creates a DataFrame with mean values
    and errors including the units for every feauer and calls the plot 
    function for them.

    Args:
        path (string): path to store the plots
        df_mean (pandas.DataFrame): DataFrame with means of all feauters
        df_stabw (pandas.DataFrame): DataFrame with standard deviation of all feauters
        properties (dictionary): dictionary with parameters for processing
    """
    params = df_mean.T.index.tolist()
    # creating dataframe with means and errors
    for param in params:
        # testen ob Einheit vorliegt
        try:
            unit = param.split()[1]
        except:
            unit = '[]'
        mean = df_mean[param]
        stabw = df_stabw[param]
        df_plot = pd.DataFrame({'mean': mean,'stabw': stabw})
        # calling plot function
        plot_mean(path, df_plot, param, unit, properties)

def plot_mean(path, df_plot, param, unit, properties):
    plot_properties = properties['plot_properties']['compare_plots']
    """
    This function plots the mean value with standard
    deviation for the given DataFrame of a property.

    Args:
        path (string): path to store the plots
        df_plot (pandas.DataFrame): Dataframe with mean and standard deviation of one feauter for all Samples
        param (string): name of the feauter
        unit (string): unit of the feauter
        properties (dictionary): dictionary with parameters for processing
    """
    colors = properties['colors_samples']
    # Create lists for the plot
    samples = df_plot.index.tolist()
    x_pos = np.arange(len(samples))
    mean = df_plot['mean']
    error = df_plot['stabw']
    # mean, error = normalizedata(mean, error)

    fig, ax = plt.subplots()
    barlist = ax.bar(x_pos, mean, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=plot_properties['label_size'])

    ### remove this for new data ###
    for sample, i in zip(samples, range(len(samples))):
        if sample == ' TNT':
            sample = 'TNT'
        barlist[i].set_color(colors[sample])
    ################################

    ytitle = 'mean ' + unit
    ax.set_ylabel(ytitle)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45)
    plt.ylabel(param.replace('_',' '), fontsize=plot_properties['label_size'])
    plt.yticks(fontsize=plot_properties['font_size'])
    plt.xticks(fontsize=plot_properties['font_size'])
    plt.tight_layout()
    # plt.show()
    save_fig(fig, path, param)
    plt.close()



def plot_features(path, properties):
    """
    This function reads and plots the mean and standard deviation files of all characteristics and samples.
    
    Args:
        path (string): root path to data
        properties (dictionary): dictionary with parameters for processing
    """
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
    plot_features(path)

