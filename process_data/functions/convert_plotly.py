# konvertieren normaler df's zu plotly geeignetem Format

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def save_fig(fig, path, name):
    # fig.tight_layout()
    # print(path)
    path = path + '\\entire_plots'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    # print(path)

    # fig.savefig(path)
    # plt.close(fig)


def convert_to_plotly(df, name, path):

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
