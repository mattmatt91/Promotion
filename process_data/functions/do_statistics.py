"""
This function calculates statistical values and performs multivariate
statistics. The values from the **results.csv** file are used for this.

A PCA and an LDA are performed. Corresponding plots are created for this.


:info: In the calculate function, parameters of the measurements can be deleted
 from the evaluation. (For example, if the same substance is used for
 all measurements. This property could be removed from the calculation)


:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""
from os import name
from tkinter import font
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from roc import get_roc
import matplotlib


def create_droplist(keywords, cols):
    """
    This function creates a list with all feauters in which the given keywords occur.

    Args:
        keywords (list): list with keywords to drop
        cols (list): list with feauters
    """
    drops = []
    for key in keywords:
        for col in cols:
            if key in col:
                drops.append(col)
    return drops


def save_df(df, path, name):
    """
    This function saves a DataFrame to csv in the results folder.

    Param:
        df (pandas.DataFrame): DataFrame to save
        path (string): path to root directory of data
        name (string): Name under which the file is to be saved
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.csv'
    df.to_csv(path, sep=';', decimal='.', index=True)


def save_html(html_object, path, name):
    """
    This Function saves a plottly figure as html to plots//statistics.

    Param:
        html_object (object): plottly html object to save
        path (string): path to root directory of data
        name (string): Name to save figure
    """
    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.html'
    print(path)
    html_object.write_html(path)


def save_jpeg(jpeg_object, path, name):
    """
    This Function saves a  figure as jpeg to plots//statistics.

    Param:
        html_object (object): plottly figure to save
        path (string): path to root directory of data
        name (string): Name to save figure
    """
    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    jpeg_object.savefig(path)


def get_statistics(df, path):
    """
    This function calculates statistical characteristics of the data.

    Param:
        df (pandas.DataFrame): DataFrame with data form result.csv
        path (string): root path to data
    """
    print('processing statistics...')
    samples = df.index.unique().tolist()
    statistics_list = {}
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    for sample in samples:
        df_stat = df[df.index == sample].describe()
        save_df(df_stat, path, sample)
        statistics_list[sample] = df_stat
        df_mean[sample] = df_stat.T['mean']
        df_std[sample] = df_stat.T['std']
    save_df(df_mean.T, path, 'mean')
    save_df(df_std.T, path, 'std')


def create_sample_list(df, properties):
    """
    This function return a list with a with all occurring 
    samples and one with the corresponding plot colours.

    Param:
        df (pandas.DataFrame): DataFrame with data form result.csv
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data

    Returns:
        color_list (list): list with sample corresponding plot colours
        sample_list (list): list with occuring samples
    """
    colors = properties['colors_samples']
    samples = df.index.unique().tolist()
    sample_list = []
    color_list = []
    for sample in samples:
        # remove this for new data
        if sample == ' TNT':
            sample = 'TNT'
        sample_list.append(sample)
        color_list.append(colors[sample])
    return color_list, sample_list


def calc_pca(df, path, df_names, properties, browser=True, dimension=True, drop_keywords=[]):
    """
    This function calculates a PCA with the data and calls the plot function. Components and
    Loadings are safed in a file in //results// folder.

    Args:
        df (pandas.DataFrame): DataFrame with data form result.csv
        path (string): root path to data
        df_names (list): names of all measurements
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
        browser (bool): With **True** the plot is created as *html* (Plottly), with **False** as *jpeg* (matplotlib)
        dimension (bool): If **True**, a *3D* plot is created, if **False**, a *2D* plot is created.
        drop_keywords (list): list with all feauters to drop before calculatin pca
    """
    print('processing pca...')
    drop_list = create_droplist(drop_keywords, df.columns)
    df.drop(drop_list, axis=1, inplace=True)
    color_list, sample_list = create_sample_list(df, properties)
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    # create df for plotting with PCs and samples as index
    df_x_pca = pd.DataFrame(x_pca, index=df.index, columns='PC1 PC2 PC3'.split())
    components = pd.DataFrame(pca.components_, columns=df.columns, index='PC1 PC2 PC3'.split())
    # plot pca
    axis_label = 'PC'
    additive_labels = [round(pca.explained_variance_ratio_[i], 2) for i in range(3)]
    plot_components(color_list, df_x_pca, sample_list, df_names, path, properties, name='PCA', browser=browser, dimension=dimension, axis_label=axis_label, additiv_labels=additive_labels)
    save_df(components, path, 'PCA_components')
    # Loadings
    process_loadings(components, path, properties)





def process_loadings(df, path, properties): # creates a df with the loadings and a column for sensor and feature
    """
    This function calculates the loadings of the PCA with the transferred components and creates a heat plot. 
    The loadings are stored in //results//.
    
    Args:
        df (pandas.DataFrame): DataFrame with Components of PCA
        path (string): root path to data
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    df_components = get_true_false_matrix(df)

    plot_loadings_heat(df_components, path, properties)
    save_df(df, path, 'PCA_loadings')


def get_true_false_matrix(df):
    """
    This function reshapes and transposes the DataFrame with the passed loadings.
    Columns with feauter and sensor are reshaped into two individual columns. 

        Args:
        df (pandas.DataFrame): DataFrame Loadings of PCA

    Returns:
        df (pandas.DataFrame): reshaped  and transposed DataFrame with Components of PCA
    """
    df = df.T
    sensors = [x[:x.find('_')] for x in df.index.tolist()]
    df['sensors'] = sensors
    features = [x[x.find('_')+1:] for x in df.index.tolist()]
    df['features'] = features
    return df


def plot_loadings_heat(df, path, properties):
    """
    This function creates a heat plot and other plots with the loadings of the PCA.

    Args:
        df (pandas.DataFrame): DataFrame with Components of PCA
        path (string): root path to data
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    # preparing dataframe
    df = convert_df_pd(df)
    df['value_abs'] = df['value'].abs()
    df['value_abs_norm'] = normalize_data(df['value_abs'])
    df['value_norm'] = normalize_data(df['value'])
    colors = [properties['sensors'][i]['color'] for i in df['sensor'].unique()]
    plot_properties = properties['plot_properties']['loadings_plot']

    # creating plot 1: total variance of the sensors per principal component
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=plot_properties['size'], dpi=plot_properties['dpi'])  # Sample figsize in inches
    ax.set_ylabel('total variance of the sensors per principal component', fontsize=plot_properties['font_size'])
    ax.set_xlabel('PC', fontsize = plot_properties['font_size'])
    sns.barplot(x="PC", y="value", data=df, ax=ax, hue='sensor', ci=None, estimator=sum, palette=colors) 
    ax.tick_params(labelsize=plot_properties['label_size'])
    ax.legend(frameon=True, fontsize=plot_properties['legend_size'])
    name = 'sensor' + '_loadings'
    save_jpeg(fig, path, name)
    # plt.show()
    plt.close()

    # creating plot 2: total variance for each sensor
    fig, ax = plt.subplots(figsize=plot_properties['size'], dpi=plot_properties['dpi'])
    sns.barplot(x="sensor", y="value_abs", data=df, ax=ax, ci=None, estimator=sum, palette=colors)
    ax.set_ylabel('total variance for each sensor', fontsize =plot_properties['font_size'])
    ax.set_xlabel('sensor', fontsize =plot_properties['font_size'])
    name = 'sensor' + '_loadings_simple'
    save_jpeg(fig, path, name)
    # plt.show()
    plt.close()
    


def convert_df_pd(df):
    """
    This function reshapes and transposes the passed DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame in

    Returns:
        df_converted (pandas.DataFrame): reshaped  DataFrame 
    """
    df.reset_index(drop=True, inplace=True)
    # formt den df um sodass pc keine Spalten mehr sind
    pcs = 'PC1 PC2 PC3'.split()
    df_converted = pd.DataFrame()
    for i, m, k in zip(df['sensors'], df['features'], range(len(df['features']))):
        for n in pcs:
            df_converted = df_converted.append({'sensor': i, 'feature': m, 'PC': n, 'value': df.iloc[k][n]}, ignore_index=True)
    return df_converted


def normalize_data(data):
    """
    This function normalises values between 0 and 1.

    Args:
        data (list): list with data

    Returns:
        data (list): list with normalised data
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calc_lda(df, path, df_names, properties, browser=True, dimension=True, drop_keywords=[]):
    """
    This function calculates a LDA with the data and calls the plot function. Plots are safed 
    in a file in //results// folder. Cross-validation is carried out (leave one out).
    The results are presented in a confusion matrix and by means of an ROC curve.

    Args:
        df (pandas.DataFrame): DataFrame with data form result.csv
        path (string): root path to data
        df_names (list): names of all measurements
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
        browser (bool): With **True** the plot is created as *html* (Plottly), with **False** as *jpeg* (matplotlib)
        dimension (bool): If **True**, a *3D* plot is created, if **False**, a *2D* plot is created.
        drop_keywords (list): list with all feauters to drop before calculatin pca
    """
    print('processing lda...')
    drop_list = create_droplist(drop_keywords, df.columns)
    df.drop(drop_list, axis=1, inplace=True)
    color_list, sample_list = create_sample_list(df, properties)
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    lda = LinearDiscriminantAnalysis(n_components=3)
    x_lda = lda.fit(scaled_data, df.index).transform(scaled_data)
    df_x_lda = pd.DataFrame(x_lda, index=df.index, columns='C1 C2 C3'.split())

    axis_label = 'C'
    plot_components(color_list, df_x_lda, sample_list, df_names, path, properties, name='LDA', browser=browser, dimension=dimension, axis_label=axis_label)
    cross_validate(lda, scaled_data, df.index, path, properties)


def cross_validate(function, x, y, path, properties):
    """
    This function performs a cross-validation (leave one out). The corresponding plot function is called. 

    Args:
        function (oobject): predictor function
        x (numpy.array): x data for learning and prediction
        y (list): y data for learning and prediction
        path (string): root path to data
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """
    plot_properties = properties['plot_properties']["confusion_matrix"]
    df_result = pd.DataFrame()
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        function.fit(x_train, y_train).transform(x_train)
        predictions = function.predict(x_test)
        result = pd.DataFrame({'true': y_test, 'predict': predictions})
        result['value'] = result['predict'] == result['true']
        df_result = df_result.append(result, ignore_index=True)
    print('error rate: ' + str((df_result[df_result['value'] == False]['value'].count()/len(df_result))*100) + '%')

    df_conf = create_confusion(df_result)
    fig, ax = plt.subplots(figsize=plot_properties['size'], dpi=plot_properties['dpi'])
    count_size = plot_properties['count_size']
    sns.heatmap(df_conf.fillna(0), linewidths=.5, annot=True, fmt='g', cbar=False, cmap="viridis", ax=ax, annot_kws={"size": count_size})
    ax.set_ylabel('true', fontsize = plot_properties['label_size'])
    ax.set_xlabel('predicted', fontsize = plot_properties['label_size'])
    plt.yticks(size=plot_properties['font_size'], rotation=30)
    plt.xticks(size=plot_properties['font_size'], rotation=30)
    plt.tight_layout()
    save_jpeg(fig, path, 'heatmap_crossvalidation_LDA')
    # plt.show()
    plt.close()
    
    # computing cor curve
    get_roc(df_result, path, properties)

def create_confusion(df):
    """
    This function creates a confusion matrix with the passed results of the cross validation.

    Args:
        df (pandas.DataFrame): DataFrame with predicted and true values from all measurements

    Returns:
        df_conf (pandas.DataFrame): DataFrame with confusion matrix. rows are true and columns predicted values
    """
    labels = df['true'].unique()
    df_conf = pd.DataFrame(columns=labels, index=labels)
    for i in df['true'].unique():
        for n in df['true'].unique():
            value = df[(df['true'] == i) & (df['predict'] == n)]['true'].count()
            df_conf.loc[i, n] = value  # zeilen sind true spalten predict
    return df_conf


def plot_components(colors, x_r, samples, df_names, path, properties, name=None, axis_label='axis', browser=False, dimension=True, additiv_labels=['', '', '']):
    """
    This function creates plots. 2D and 3D plots can be created. These are created with
    matplotlib or plottly. They can be saved as jpeg or html.

    Args:
        colors (list): list with all colours that occur
        x_r (numpy.array): x data to plot
        samples (list): list with all samples that occur
        df_names (list): names of all measurements 
        path (string): root path to data
        name (string): name to save the plot, default is *None*
        axis_label (string): label for axis, default is **axis**
        browser (bool): With **True** the plot is created as *html* (Plottly), with **False** as *jpeg* (matplotlib)
        dimension (bool): If **True**, a *3D* plot is created, if **False**, a *2D* plot is created.
        additiv_labels (list): list with additiv labels for each axis, , default is empty
    """ 
    if not browser:
        if dimension:
            plot_properties = properties['plot_properties'][ "components_plot_3D"]
            matplotlib.rcParams['legend.fontsize'] = plot_properties['legend_size']
            fig = plt.figure(figsize=plot_properties['size'], dpi=plot_properties['dpi'])   
            threedee = fig.add_subplot(111, projection='3d')
            for color, target_name in zip(colors, samples):
                threedee.scatter(x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(1)],
                                 x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(2)],
                                 x_r.loc[x_r.index.get_level_values('sample') == target_name][axis_label + str(3)],
                                 s= plot_properties['dot'], color=color, alpha=.8, label=target_name)

            threedee.tick_params(axis='both', labelsize=plot_properties['font_size'])
            threedee.legend(loc='best', shadow=False, scatterpoints=1)
            threedee.set_xlabel('{0}{1} {2} %'.format(axis_label, 1, additiv_labels[0]),fontsize=plot_properties['label_size'])
            threedee.set_ylabel('{0}{1} {2} %'.format(axis_label, 2, additiv_labels[1]),fontsize=plot_properties['label_size'])
            threedee.set_zlabel('{0}{1} {2} %'.format(axis_label, 3, additiv_labels[2]),fontsize=plot_properties['label_size'])
            save_jpeg(fig, path, name+'_3D')

        # 2D-Plot
        if not dimension:
            plot_properties = properties['plot_properties'][ "components_plot_2D"]
            matplotlib.rcParams['legend.fontsize'] = plot_properties['legend_size']
            fig = plt.figure(figsize=plot_properties['size'], dpi=plot_properties['dpi'])
            twodee = fig.add_subplot()
            for color, i, target_name in zip(colors, np.arange(len(colors)), samples):
                twodee.scatter(x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(1)],
                               x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(2)],
                               s=plot_properties['dot'], color=color, alpha=.8, label=target_name)

            twodee.tick_params(axis='both', labelsize=plot_properties['font_size'])
            twodee.legend(loc='best', shadow=False, scatterpoints=1) 
            twodee.set_xlabel('{0}{1} {2} %'.format(axis_label, 1, additiv_labels[0]),fontsize=plot_properties['label_size'])
            twodee.set_ylabel('{0}{1} {2} %'.format(axis_label, 2, additiv_labels[1]),fontsize=plot_properties['label_size'])
            save_jpeg(fig, path, name+'_2D')
            
        # plt.show()  
        
    if browser:
        plot_properties = properties['plot_properties'][ "components_plot_html"]
        axis_names = [axis_label+ str(i+1) for i in range(3)]

        colors_dict = {}
        for i in x_r.index.unique():
            colors_dict[i]=properties['colors_samples'][i]
        fig = px.scatter_3d(
                x_r,
                x=axis_names[0],
                y=axis_names[1],
                z=axis_names[2],
                color_discrete_map=colors_dict,
                color=x_r.index,
                hover_data={'name': df_names.tolist()}
                )

        # setting plot parameters
        fig.update_layout(
            legend_title_font_size=plot_properties['legend_size'],
            legend_font_size=plot_properties['legend_size']/1.2,
            font_size=plot_properties['font_size']
            )

        # saving plot
        save_html(fig, path, name)
        # fig.show()
    plt.close()


def calculate(path, properties, statistic=True, pca=True, lda=True, browser=False, dimension=False):
    """
    This is the main function of this module. It calculates some statistics, LDA and PCA. Plots are created 

    Args:
        path (string): root path to data
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
        pca (bool): If True, PCA is executed (*default*)
        lda (bool): If True, LDA is executed (*default*)
        browser (bool): With **True** the plot is created as *html* (Plottly), with **False** as *jpeg* (matplotlib)
        dimension (bool): If **True**, a *3D* plot is created, if **False**, a *2D* plot is created.
    """
    # preparing result.csv for statistics
    path = path + '\\Results'
    path_results = path+ '\\Result.csv'
    df = pd.read_csv(path_results, delimiter=';', decimal=',')
    df_names = df['name']
    df.drop(['rate', 'duration', 'sensors', 'path', 'droptime', 'height',
       'sample_number', 'info', 'datetime', 'name'], axis=1, inplace=True)# select sensors to drop for statisctics e.g. name
    df.set_index(['sample'], inplace=True)

    # do statistics
    if statistic:
        get_statistics(df, path)
    if pca:
        calc_pca(df, path, df_names, properties, browser=browser, dimension=dimension, drop_keywords=[])
    if lda:
        calc_lda(df, path, df_names, properties, browser=browser, dimension=dimension, drop_keywords=[])
    


if __name__ == '__main__':
    path = 'E:\\Promotion\Daten\\29.06.21_Paper_reduziert'
    calculate(path)
