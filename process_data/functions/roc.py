"""
This module computes the ROC curve and plots it.

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path
import json as js
from sklearn.metrics import roc_auc_score
import matplotlib


def read_json(filename):
    """
    This function reads a json file  in the current directory and returns the corresponding string.

    Args:
        filename  (string): name of the json file  

    Returns:
        jsonstring (string): json file
    """
    with open(filename) as json_file:
        return js.load(json_file)


def extract_properties():
    """
    This function reads the properties file and returns it as a dictionary
    the file contains every important information about the data processing.

    Returns:
        jsonstring (dictionary): dictionary with properties
    """
    path = str(Path().absolute()) + '\\properties.json'
    print(path)
    dict = read_json(path)
    return dict


def save_jpeg(jpeg_object, path, name):
    """
    This function saves a figure as jpeg.
    directory is root//plot//statistics.

    Args:
        jpeg_object (object): name of the json file
        path (string): root path of data
        name (string): name of the measurement
    """
    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    jpeg_object.savefig(path)


def get_roc(df, path, properties):
    """
    This computes ROC values and plots it.

    Args:
        df (pandas.DataFrame): Dataframe with true and predicted values
        path (string): root path for storing plots
        properties (dictionary): dictionary with all important properties
    """
    colors = properties['colors_samples']
    samples = df['true'].unique()
    dict_samples = {}
    for sample, i in zip(samples, range(len(samples))):
        dict_samples[sample] = i

    classes = [dict_samples[i] for i in dict_samples]

    #####################################################################
    # binarize sample names to bool array

    # In most examples, the predicted data 
    # had floats as the format. ([0.1 0.2 0.7]). 
    # However, I only get integers ([0 0 1] ) 
    # since mit classifier returns the class
    # with the predict function of the lda. 
    # Does it make sense to work with this format?

    true_values = np.array([dict_samples[i] for i in df['true']])
    true_values = label_binarize(true_values, classes=classes)
    predict_values = np.array([dict_samples[i] for i in df['predict']])
    predict_values = label_binarize(predict_values, classes=classes)
    
    #####################################################################

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in classes:
        fpr[i], tpr[i], _ = roc_curve(true_values[:, i], predict_values[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_values.ravel(), predict_values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    macro_roc_auc_ovo = roc_auc_score(true_values, predict_values, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(true_values, predict_values, multi_class="ovo", average="weighted")
    macro_roc_auc_ovr = roc_auc_score(true_values, predict_values, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(true_values, predict_values, multi_class="ovr", average="weighted")

    print("ROC AUC scores:\n{:.6f} (macro),\n{:.6f} ""(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    print("ROC AUC scores:\n{:.6f} (macro),\n{:.6f} ""(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    ## Plot all ROC curves ## 
    plot_properties = properties['plot_properties']['roc_plot']
    lw = 2
    fig = plt.figure(figsize=[10,10])
    plt.yticks(size=plot_properties['font_size'], rotation=30)
    plt.xticks(size=plot_properties['font_size'], rotation=30)
    

    # plot micro
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average (area = {0:0.2f})".format(roc_auc["micro"]),
        color="darkred",
        linestyle=":",
        linewidth=4,
    )

    # plot macros
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average (area = {0:0.2f})".format(roc_auc["macro"]),
        color="darkblue",
        linestyle=":",
        linewidth=4,
    )

    # plot roc for each sample
    for i in range(len(classes)):
        sample = samples[i]
        plt.plot(
            fpr[i],
            tpr[i],
            colors[sample],
            lw=lw,
            label="{0} (area = {1:0.2f})".format(samples[i], roc_auc[i]),
        )

    # plot diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    matplotlib.rcParams['legend.fontsize'] = plot_properties['legend_size']
    plt.xlim([-.01, 1.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize = plot_properties['label_size'])
    plt.ylabel("True Positive Rate", fontsize = plot_properties['label_size'])
    plt.legend(loc=0)
    plt.tight_layout()
    save_jpeg(fig, path, 'roc')
    # plt.show()
    plt.close()

def read_roc(path):
    """
    This function computes roc of a local file, as standalone script.

    Args:
        path (string): path to file with true predict values
    """
    properties = extract_properties()
    df = pd.read_csv(path)
    try:
        df.drop('Unnamed: 0', inplace=True, axis=1)
    except:
        pass
    # print(df)
    get_roc(df, '', properties)
    plt.show()

if __name__ == '__main__':
    read_roc('roc1.txt')
    
