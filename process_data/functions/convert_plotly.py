"""

This module converts datasets into a format that can be used by plottly..

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def save_fig(fig, path, name):
    """
    This function saves a figure to a file.

    Args:
        fig  (object): Figure to save
        path (string): path to root directory, figure is saved in root/entire_plots
        name (string): name for the saved file  
    """
    # fig.tight_layout()
    # print(path)
    path = path + '\\entire_plots'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    # print(path)

    # fig.savefig(path)
    # plt.close(fig)


def convert_to_plotly(df, name, path):
    """
    This function transforms data from satadard format to a plottly suitable format.

    Args:
        df  (pandas.DataFrame): Dataframe to convert
        name (string): name for the data
        path (string): root path to save the plot
    """
    print(df)
    samples = df.columns
    df['wavelength']= df.index
    df = pd.melt(df, id_vars='wavelength', value_vars=df.columns[:-1])
    df['counts'] = df['value']
    # print(df)
    fig = px.line(df, x='wavelength', y='counts', color='variable')
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    fig.show()
    # save_fig(fig, path, name)
