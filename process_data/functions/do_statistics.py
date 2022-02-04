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



def get_colors(i):
    """
    bkbadfadsf
    """
    colors = []
    for n in range(i):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors


def save_string(string, path, name):
    path = path[:path.rfind('\\')]
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.txt'
    text_file = open(path, "w")
    for s in string:
        text_file.write(s)
    text_file.close()


def create_droplist(keywords, cols):
    drops = []
    for key in keywords:
        for col in cols:
            if key in col:
                drops.append(col)
    return drops


def save_df(df, path, name):
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.csv'
    df.to_csv(path, sep=';', decimal='.', index=True)


def save_html(html_object, path, name):
    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.html'
    print(path)
    html_object.write_html(path)


def save_jpeg(jpeg_object, path, name):
    path = path + '\\plots\\statistics'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    jpeg_object.savefig(path)


def get_statistics(df, path):
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


def create_sample_dict(df, properties):
    colors = properties['colors']
    samples = df.index.unique().tolist()
    sample_list = []
    color_list = []
    for sample in samples:
        if sample == ' TNT':
            sample = 'TNT'
        sample_list.append(sample)
        color_list.append(colors[sample])
    return color_list, sample_list


def calc_pca(df, path, df_names, properties, browser=True, dimension=True, drop_keywords=[]):
    print('processing pca...')
    drop_list = create_droplist(drop_keywords, df.columns)
    df.drop(drop_list, axis=1, inplace=True)
    color_list, sample_list = create_sample_dict(df, properties)
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    # create df for plotting with PCs and samples as index
    df_x_pca = pd.DataFrame(x_pca, index=df.index, columns='PC1 PC2 PC3'.split())
    # plot pca
    axis_label = 'PC'
    additive_labels = [round(pca.explained_variance_ratio_[i], 2) for i in range(3)]
    plot_components(color_list, df_x_pca, sample_list, df_names, path, name='PCA', browser=browser, dimension=dimension, axis_label=axis_label, additiv_labels=additive_labels)
    # Loadings
    process_loadings(pd.DataFrame(pca.components_, columns=df.columns, index='PC1 PC2 PC3'.split()), path)





def process_loadings(df, path): # creates a df with the loadings and a column for sensor and feature
    df_components = get_true_false_matrix(df)

    plot_loadings_heat(df_components, path)
    save_df(df, path, 'PCA_loadings')


def get_true_false_matrix(df):
    df = df.T
    sensors = [x[:x.find('_')] for x in df.index.tolist()]
    df['sensors'] = sensors
    features = [x[x.find('_')+1:] for x in df.index.tolist()]
    df['features'] = features
    return df


def plot_loadings_heat(df, path):
    df = convert_df_pd(df)
    df['value_abs'] = df['value'].abs()
    df['value_abs_norm'] = normalize_data(df['value_abs'])
    df['value_norm'] = normalize_data(df['value'])
    sns.color_palette("viridis", as_cmap=True)
    sns.set_theme()
    
    # hier werden einige Plots zu den Loadings erstellt
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches
    sns.barplot(x="PC", y="value", data=df, ax=ax, hue='sensor', ci=None, estimator=sum)
    # ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('total variance of the sensors per principal component')
    name = 'sensor' + '_loadings'
    save_jpeg(fig, path, name)
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches
    sns.barplot(x="sensor", y="value_abs", data=df, ax=ax, ci=None, estimator=sum)
    # ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('total variance for each sensor')
    name = 'sensor' + '_loadings_simple'
    save_jpeg(fig, path, name)
    plt.close()
    plt.show()


def convert_df_pd(df):
    df.reset_index(drop=True, inplace=True)
    # formt den df um sodass pc keine Spalten mehr sind
    pcs = 'PC1 PC2 PC3'.split()
    df_new = pd.DataFrame()
    for i, m, k in zip(df['sensors'], df['features'], range(len(df['features']))):
        for n in pcs:
            df_new = df_new.append({'sensor': i, 'feature': m, 'PC': n, 'value': df.iloc[k][n]}, ignore_index=True)
    return df_new


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



def calc_lda(df, path, df_names, properties, browser=True, dimension=True, drop_keywords=[]):
    print('processing lda...')
    drop_list = create_droplist(drop_keywords, df.columns)
    df.drop(drop_list, axis=1, inplace=True)
    color_list, sample_list = create_sample_dict(df, properties)
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    lda = LinearDiscriminantAnalysis(n_components=3)
    x_lda = lda.fit(scaled_data, df.index).transform(scaled_data)
    df_x_lda = pd.DataFrame(x_lda, index=df.index, columns='C1 C2 C3'.split())

    axis_label = 'C'
    plot_components(color_list, df_x_lda, sample_list, df_names, path, name='LDA', browser=browser, dimension=dimension, axis_label=axis_label)
    cross_validate(lda, scaled_data, df.index, path, properties)

def cross_validate(function, x, y, path, properties):
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
    sns.set(font_scale=1.5)
    get_roc(df_result, path, properties)
    df_conf = create_confusion(df_result)
    fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches
    sns.heatmap(df_conf.fillna(0), linewidths=.5, annot=True, fmt='g', cbar=False, cmap="viridis", ax=ax)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_jpeg(fig, path, 'heatmap_crossvalidation_LDA')
    # plt.show()


def create_confusion(df):
    labels = df['true'].unique()
    df_conf = pd.DataFrame(columns=labels, index=labels)
    for i in df['true'].unique():
        for n in df['true'].unique():
            value = df[(df['true'] == i) & (df['predict'] == n)]['true'].count()
            df_conf.loc[i, n] = value  # zeilen sind true spalten predict
    return df_conf


def plot_components(colors, x_r, samples, df_names, path, name=None, axis_label='axis', browser=False, dimension=True, additiv_labels=['', '', '']):
    if not browser:
        if dimension:
            fig = plt.figure()
            threedee = fig.add_subplot(111, projection='3d')
            for color, target_name in zip(colors, samples):
                threedee.scatter(x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(1)],
                                 x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(2)],
                                 x_r.loc[x_r.index.get_level_values('sample') == target_name][axis_label + str(3)],
                                 s=30, color=color, alpha=.8, label=target_name)
            threedee.legend(loc='best', shadow=False, scatterpoints=1)
            threedee.set_xlabel('{0}{1} {2} %'.format(axis_label, 1, additiv_labels[0]))
            threedee.set_ylabel('{0}{1} {2} %'.format(axis_label, 2, additiv_labels[1]))
            threedee.set_zlabel('{0}{1} {2} %'.format(axis_label, 3, additiv_labels[2]))

        # 2D-Plot
        if not dimension:
            twodee = plt.figure().add_subplot()
            for color, i, target_name in zip(colors, np.arange(len(colors)), samples):
                twodee.scatter(x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(1)],
                               x_r[x_r.index.get_level_values('sample') == target_name][axis_label + str(2)],
                               s=30, color=color, alpha=.8, label=target_name)
            twodee.legend(loc='best', shadow=False, scatterpoints=1) 
            twodee.set_xlabel('{0}{1} {2} %'.format(axis_label, 1, additiv_labels[0]))
            twodee.set_ylabel('{0}{1} {2} %'.format(axis_label, 2, additiv_labels[1]))

        plt.show()    
    if browser:
        axis_names = [axis_label+ str(i+1) for i in range(3)]
        fig = px.scatter_3d(x_r, x=axis_names[0], y=axis_names[1], z=axis_names[2], color=x_r.index.get_level_values('sample'),
                            hover_data={'name': df_names.tolist()})
        # fig.show()
        save_html(fig, path, name)
    plt.close()


def calculate(path, properties, statistic=True, pca=True, lda=True, svm=False, knn=False, browser=False, dimension=False):
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
