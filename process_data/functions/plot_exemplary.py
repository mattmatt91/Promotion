
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from read_files import extract_properties


def save_fig(fig, path, name):
    fig.tight_layout()
    # print(path)
    path = path + '\\entire_plots'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.jpeg'
    # print(path)
    # plt.show()
    fig.savefig(path)
    plt.close(fig)

def plot_ex(df, path, sensor, x_lim_plot, colors):

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
    print(path)
    df = pd.read_csv(path, decimal=',', sep='\t')
    df.set_index('time [s]', inplace=True)
    print(sensor)
    plot_ex(df, root_path, sensor, x_lim_plot, colors)


def main(root_path): # diese Dateien müssen manuell ertellt werden
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
    root_path = 'C:\\Users\\Matthias\\Desktop\\Messaufbau\\dataaquisition\\data\\test_auto\\results\\plots\\param_plots'
    main(root_path)