"""
This module reads additional data from the specrtometer
from the *filenae*_spectra.json file and returns it as a dictionaty

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
:license: see LICENSE for more details.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def plot_spectra(data, properties, path):
    # creating fig
    plot_properties = properties['plot_properties']['Spectrometer_plot']
    fig, ax = plt.subplots(sharex=True, dpi=plot_properties['dpi'], figsize=plot_properties['size'])
    
    # plotting data
    ax.plot(data['dif']) #  color=self.properties['sensors'][sensor]['color'])  
    
    # setting up labels
    ax.set_ylabel('Intensity [counts]', rotation=90, fontsize = plot_properties['label_size'])
    ax.set_xlabel("wavelength [nm]" , fontsize=plot_properties['label_size'])
    
    # setting up ticks
    ax.tick_params(axis='y', labelsize= plot_properties['font_size'])
    ax.tick_params(axis='x', labelsize= plot_properties['font_size'])
    
    # optimizing figure
    ax.grid()
    fig.tight_layout()
    
    # creating path
    name = path[path.rfind('\\')+1:]
    path = path[:path.rfind('\\')] + '\\results\\plots\\single_measurements' 
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '_spectra.jpeg'
    fig.tight_layout()
    # plt.show()
    fig.savefig(path)
    plt.close(fig)
    

def get_info(data):
    results = {}
    results['Spectrometer_max'] = data['dif'].max()
    results['Spectrometer_integral'] = np.trapz(data['dif'])
    return results
    

def read_file(path):
    data = pd.read_csv(path, delimiter='\t', decimal='.', dtype=float)
    data.set_index(data.columns[0], inplace=True)
    return data.abs()


def read_spectra(path, properties):
    """
    This function reads the file with information about the spectra (*filenae*_spectra.json)
    and returns it.

    Args:
        path (string): path to the folder of the measurement
    """
    path_folder = path + path[path.rfind('\\'):]
    path_file = path_folder + '_spectrometer.txt'
    data = read_file(path_file)
    results = get_info(data)
    plot_spectra(data, properties, path)
    return results
