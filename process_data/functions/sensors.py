"""
functions.sensors
-----------------

This script exrtacts features from the data, saves the feauters 
from all measurements to global results file and creates 
one file for every sensor with all measurements.

:copyright: (c) 2021 by Matthias Muhr, Hochschule Bonn Rhein Sieg
:license: see LICENSE for more details.
"""

from pyexpat import features
import pandas as pd
from scipy.signal import chirp, find_peaks, peak_widths
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



######################################################################################################################
## this class creates objects for every sensor and stores all measurment for this sensor ##
class Sensor:
    """This is for creating objects for every sensor and stores the data from all measurements 
    in this object. Sensor names are picked up in properties. One DataFrame for every sensor is created

    :param properties: properties is a dictionary with all parameters for evaluating the data
    :type properties: dict
    """
    
    def __init__(self, properties):
        """
        constructor method
        """
        df_dict = {} # dicht with df for each sensor, one for all measurments
        self.properties = properties
        for sensor in self.properties['sensors']:
            df_dict[sensor] = pd.DataFrame()
        self.df_dict = df_dict


    def add_item(self,df, name): # append data from measurement in global sensor df
        """This function sorts the passed DataFrame into those of the sensor 
        object and names the respective columns with the name of the measurement.

        :param df: The columns of the DataFrame should match those in the properties.json file.
        :type df: pandas DataFrame

        :param name: Measurement name 
        :type name: string
        """
        for sensor in self.properties['sensors']:
            self.df_dict[sensor][name] = df[sensor]


    def save_items(self, path): # save one file for every sensor with all measurements
        """This function saves all DataFrames contained in the sensor object, one file 
        is saved per sensor. A folder "results" is created in the root folder where 
        the files are stored.

        :param path: Path to the folder in which the measurement folders are stored
        :type path: string
        """
        for sensor in self.properties['sensors']:
            name = sensor + '_gesamt'
            save_df(self.df_dict[sensor], path, name)


class Plot:
    """
    This class creates plot objects. For each one, an image with all sensors of a measurement is created.

    :param name: Name of the measurement
    :type name: string

    :param size: Number of sensors to be plotted
    :type size: int
    """
    def __init__(self,name, size):
        """
        constructor method
        """
        self.fig, self.axs = plt.subplots(size, sharex=True, figsize=(14, 8))
        self.name = name
        #self.fig.suptitle(name)
        self.i = 0

    def add_subplot(self, sensor, df_corr, properties, results_half, results_full, peaks):
        """This function assigns a subplot for the corresponding sensor to the plot object

        :param sensor: Name of the sensor
        :type sensor: string

        :param df_corr: Dataframe with prepared data from measurement
        :type sensor: pandas Dataframe
        
        :param properties: properties is a dictionary with all parameters for evaluating the data
        :type properties: dict

        :param results_half: Array with from measurement extracted feauters for the half peak
        :type results_half: numpy array

        :param results_full: Array with from measurement extracted feauters for the full peak
        :type results_full: numpy array

        :param peaks: Array with from measurement extracted feauters for detected peaks
        :type peaks: numpy array
        """
        self.axs[self.i].plot(df_corr[sensor])
        ## print peaks in plot
        if peaks.size != 0:
            self.axs[self.i].plot(df_corr.index[peaks], df_corr[sensor][df_corr.index[peaks]], "x")
            self.axs[self.i].vlines(x=df_corr.index[peaks][0], ymin=df_corr[sensor][df_corr.index[peaks][0]] - properties["prominences"],
                       ymax=df_corr[sensor][df_corr.index[peaks][0]], color="C1")
            self.axs[self.i].hlines(y=properties["width_heights"], xmin=df_corr.index[int(properties["left_ips"])],
                       xmax=df_corr.index[int(properties["right_ips"])], color="C1")
            self.axs[self.i].hlines(y=results_full[1], xmin=df_corr.index[int(results_full[2])],
                       xmax=df_corr.index[int(results_full[3])],
                       color="C2")
            self.axs[self.i].hlines(y=results_half[1], xmin=df_corr.index[int(results_half[2])],
                       xmax=df_corr.index[int(results_half[3])],
                       color="C2")
        # self.axs[self.i].set_title(sensor)
        self.axs[self.i].set(xlabel='time [s]', ylabel=sensor + ' [V]')
        self.axs[self.i].grid()
        try:
            self.axs[self.i].set_yticks(np.arange(0,np.max(df_corr[sensor]),np.max(df_corr[sensor])/3))
        except:
            self.axs[self.i].set_yticks(np.arange(0,5,5/3))
        self.i = self.i +1

    def show_fig(self, path):
        """This function saves the created plot object in the folder "results\\plots\\single_measurements".

        :param path: Path to the folder in which the measurement folders are stored
        :type path: string
        """
        self.fig.tight_layout()
        path = path + '\\results\\plots\\single_measurements'
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + self.name + '.jpeg'
        # plt.show()
        self.fig.savefig(path)
        plt.close(self.fig)


def width_clip(x, threshold):
    x = x.tolist()
    flag = False
    list_peaks = []
    start = 0
    end = 0
    for i in range(len(x)):
        if flag == False and x[i] > threshold:
            flag = True
            start = i
        elif flag == True and x[i] < threshold:
            flag = False
            end = i
            list_peaks.append(end-start)
    if len(list_peaks) == 0 or np.max(list_peaks) <= 4:
        return 0
    else:
        # print(list_peaks)
        return np.max(list_peaks)

def running_mean(x):
    N = 20 # über wie viele Werte wird geglättet
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

## get slope of a time section ##
def get_slope(x,t):
    end = 0
    flag = False
    for i in range(len(x)-1):
        if flag == False:
            if x[i+1] > x[i]:
                pass
            else:
                end = i
                flag = True
    slope = (x[end]-x[0])/(t[end]-t[0])
    return slope


##  saving the result df ##
def save_df(df, path, name):
    path = path + '\\results'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.csv'
    print(name + 'saved as ' + path)
    df.to_csv(path, sep=';', decimal=',', index = True)



def evaluate_sensor(df, sensor, threshold):
    peaks, properties = find_peaks(df[sensor], prominence=0, width=1, distance=20000, height=threshold)
    results_half = peak_widths(df[sensor], peaks, rel_height=0.5)
    results_full = peak_widths(df[sensor], peaks, rel_height=0.99)
    try:
        df_peak = pd.DataFrame(df[sensor].iloc[int(results_full[2]):int(results_full[3])])
    except:
        pass
    # functions for feature extraction
    def get_peak():
         return df.index[int(peaks[0])]
    def get_start():
        return df.index[int(results_full[2])]
    def get_stop():
        return df.index[int(results_full[3])]
    def get_width():
        return df.index[int(results_full[3])] - df.index[int(results_full[2])]
    def get_width_half():
        return df.index[int(results_half[3])] - df.index[int(results_half[2])]
    def get_height():
        return df[sensor][df.index[int(peaks[0])]]
    def get_integral():
        return np.trapz(df_peak[sensor] ,x=df_peak.index)
    def get_slope_2():
        return get_slope(df_peak[sensor].tolist(), df_peak.index.tolist())
    def get_width_clip():
        return width_clip(df[sensor], 4.9)
    def get_width_heigth():
        return (df.index[int(results_full[3])] - df.index[int(results_full[2])])/(df[sensor][df.index[int(peaks[0])]])

    values = [get_peak, get_start, get_stop,get_width, get_width_half, get_height, get_integral, get_slope_2, get_width_clip, get_width_heigth]
    features = "peak[s] start[s] stop[s] width[s] width_half[s] height intetegral[Vs] slope[V/s] width_clip[s] width_heigth[s/V]".split()
    
    #build the json result for this measurement
    result_dict = {}
    for feature, value in zip(features,values):
        name = "{0}_{1} {2}".format(sensor, feature[:feature.find('[')], feature[feature.find('['):])
        try:
            result_dict[name] = value()
        except:
            result_dict[name] = 0

    return (peaks, properties, results_half, results_full, result_dict)


def cut_peakarea(df, sensor_to_cut,sensors_norm):
    place_before_peak = 1000
    place_after_peak = 10000
    step = 0.00001
    len_signal = step * (place_after_peak + place_before_peak)
    # cuts the important part of the file, adds running mean col and ammount of signals
    try:
        # error = 1/0
        index_sensor_to_cut_max = df[sensor_to_cut].idxmax(axis = 0)
        if index_sensor_to_cut_max <= place_before_peak:
            index_sensor_to_cut_max = place_before_peak
        elif index_sensor_to_cut_max >= (len(df.index)- place_after_peak):
            index_sensor_to_cut_max = len(df.index)- place_after_peak
    except:
        print('no maximum found')
        index_sensor_to_cut_max = len(df.index)//2
    
    df_corr = df.iloc[index_sensor_to_cut_max - place_before_peak:index_sensor_to_cut_max + place_after_peak].apply(np.abs)
    df_corr['time [s]'] = np.arange(0, 0.11, 0.00001)
    for sensor in sensors_norm:
        df_corr[[sensor + '_nm']] = df_corr[[sensor]].apply(running_mean)
        # df_corr.drop(sensor, axis=1, inplace=True)
    df_corr.set_index('time [s]', inplace=True)
    return df_corr

def read_file(path,decimal,name, path_out, object_raw, properties):
    
    threshold = properties['threshold']
    path = path + path[path.rfind('\\'):] + '.txt'
    dict_result = {}
    df_measurement = pd.read_csv(path, delimiter='\t', decimal=decimal, dtype=float)
    df_corr = cut_peakarea(df_measurement, properties['sensor_to_cut'], properties['sensors_norm'])
    object_raw.add_item(df_corr, name) # adding data from measurement to df for each sensor including all measurements
    fig = Plot(name,len(df_corr.columns))
    for this_sensor in df_corr.columns:
        peaks, properties, results_half, results_full, this_dict_result = evaluate_sensor(df_corr, this_sensor, threshold[this_sensor])
        dict_result.update(this_dict_result)
        fig.add_subplot(this_sensor, df_corr, properties, results_half, results_full, peaks)
    fig.show_fig(path_out)
    return dict_result



