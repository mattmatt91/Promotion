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


class Spectra:
    """This is for creating objects for every sensor and stores the data from all measurements 
    in this object. Sensor names are picked up in properties. One DataFrame for every sensor is created

    Args:
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """

    def __init__(self, properties):
        """
        constructor method
        """
        self.properties = properties
        self.data= pd.DataFrame()


    def add_item(self,df, name): # append data from measurement in global sensor df
        """This function sorts the passed DataFrame into those of the sensor 
        object and names the respective columns with the name of the measurement.

        Args:
            df (pandas.DataFrame): The columns of the DataFrame should match those in the properties.json file.
            name (string): Measurement name 
        """
        self.data[name] = df[['dif']]
        print(self.data)


    def save_items(self, path): # save one file for every sensor with all measurements
        """This function saves all DataFrames contained in the sensor object, one file 
        is saved per sensor. A folder "results" is created in the root folder where 
        the files are stored.

        Args:
            path (string): Path to the folder in which the measurement folders are stored
        """
        name = 'specta_gesamt'
        self.save_df(self.data, path, name)
        


    def save_df(self, path): 
        """
        his function saves a DataFrame to csv in the results folder.

        Param:
            df (pandas.DataFrame): DataFrame to save
            path (string): path to root directory of data
            name (string): Name under which the file is to be saved
        """
        path = path + '\\results'
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\spectra_gesamt.csv'
        print(name + 'saved as ' + path)
        self.data.to_csv(path, sep=';', decimal=',', index = True)


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


def read_spectra(path, properties, spectra_object, name):
    """
    This function reads the file with information about the spectra (*filenae*_spectra.json)
    and returns it.

    Args:
        path (string): path to the folder of the measurement
    """
    path_folder = path + '\\' + name
    path_file = path_folder + '_spectra.txt'
    data = read_file(path_file)
    spectra_object.add_item(data, name)
    results = get_info(data)
    plot_spectra(data, properties, path)
    return results
