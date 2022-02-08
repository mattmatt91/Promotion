"""
This module exrtacts features from the data, saves the feauters 
from all measurements to global results file and creates 
one file for every sensor with all measurements.

:copyright: (c) 2022 by Matthias Muhr, Hochschule-Bonn-Rhein-Sieg
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

    Args:
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
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

        Args:
            df (pandas.DataFrame): The columns of the DataFrame should match those in the properties.json file.
            name (string): Measurement name 
        """
        sensors_to_delelte = []
        for sensor in self.df_dict:    
            try:
                self.df_dict[sensor][name] = df[sensor]
            except:
                print('no data for {0} in {1}'.format(sensor, name))
                sensors_to_delelte.append(sensor)
        # deleting dataframes for sensors not included in measurements
        for del_sensor in sensors_to_delelte:
            del self.df_dict[del_sensor]
            print('deleting {} from results'.format(del_sensor))


    def save_items(self, path): # save one file for every sensor with all measurements
        """This function saves all DataFrames contained in the sensor object, one file 
        is saved per sensor. A folder "results" is created in the root folder where 
        the files are stored.

        Args:
            path (string): Path to the folder in which the measurement folders are stored
        """
        for sensor in self.properties['sensors']:
            name = sensor + '_gesamt'
            save_df(self.df_dict[sensor], path, name)


######################################################################################################################
## this class creates plots with all sensors for every measurement ##
class Plot:
    """
    This class creates plot objects. For each one, an image with all sensors of a measurement is created.

    Args:
        name (string): Name of the measurement
        size (int): Number of sensors to be plotted
        properties (dictionary): properties is a dictionary with all parameters for evaluating the data
    """

    def __init__(self,name, size, properties):
        """
        constructor method
        """
        self.plot_properties = properties['plot_properties']['measurement_plot']
        self.properties = properties
        self.fig, self.axs = plt.subplots(size, sharex=True, dpi=self.plot_properties['dpi'], figsize=self.plot_properties['size'])
        self.name = name
        self.i = 0


    def add_subplot(self, sensor, df_corr, peak_properties, results_half, results_full, peaks):
        """This function assigns a subplot for the corresponding sensor to the plot object.

        Args:
            sensor (string): Name of the sensor
            df_corr (pandas.DataFrame): Dataframe with prepared data from measurement
            peak_properties (dictionary): peak_properties is a dictionary with data about extracted peaks
            results_half (numpy.array): Array with from measurement extracted feauters for the half peak
            results_full (numpy.array): Array with from measurement extracted feauters for the full peak
            peaks (numpy.array): Array with from measurement extracted feauters for detected peaks
        """
        self.axs[self.i].plot(df_corr[sensor], color=self.properties['sensors'][sensor]['color'])
        ## print peaks in plot
        if peaks.size != 0:
            self.axs[self.i].plot(df_corr.index[peaks], df_corr[sensor][df_corr.index[peaks]], "x")
            self.axs[self.i].vlines(x=df_corr.index[peaks][0], ymin=df_corr[sensor][df_corr.index[peaks][0]] - peak_properties["prominences"],
                       ymax=df_corr[sensor][df_corr.index[peaks][0]], color="C1")
            self.axs[self.i].hlines(y=peak_properties["width_heights"], xmin=df_corr.index[int(peak_properties["left_ips"])],
                       xmax=df_corr.index[int(peak_properties["right_ips"])], color="C1")
            self.axs[self.i].hlines(y=results_full[1], xmin=df_corr.index[int(results_full[2])],
                       xmax=df_corr.index[int(results_full[3])],
                       color="C2")
            self.axs[self.i].hlines(y=results_half[1], xmin=df_corr.index[int(results_half[2])],
                       xmax=df_corr.index[int(results_half[3])],
                       color="C2")

        label = sensor + ' [V]'
        self.axs[self.i].set_ylabel(label, rotation=0, loc='top', fontsize = self.plot_properties['label_size'])
        self.axs[self.i].tick_params(axis='y', labelsize= self.plot_properties['font_size'])
        self.axs[self.i].grid()
        try:
            self.axs[self.i].set_yticks(np.arange(0,np.max(df_corr[sensor]),round(np.max(df_corr[sensor])/3, 2)))
        except:
            self.axs[self.i].set_yticks(np.arange(0,5,5/3))
        self.i = self.i +1

    def show_fig(self, path):
        """This function saves the created plot object in the folder "results\\plots\\single_measurements".

        Args:
            path (string): Path to the folder in which the measurement folders are stored
        """
        self.axs[-1].set_xlabel("time [s]" , fontsize = self.plot_properties['label_size'])
        plt.xticks(fontsize=self.plot_properties['font_size'])
        self.axs[-1].get_shared_x_axes().join(*self.axs)
        self.fig.tight_layout()
        path = path + '\\results\\plots\\single_measurements'
        Path(path).mkdir(parents=True, exist_ok=True)
        path = path + '\\' + self.name + '.jpeg'
        
        self.fig.tight_layout()
        # plt.show()
        self.fig.savefig(path)
        plt.close(self.fig)


def width_clip(x, threshold):
    """This function extracts the feauter "width clip", which calculates the length at which a signal is too large for the measuring range.
        
        Args:
            x (list): Time series from which the feature is to be extracted
            threshold (float): Value from which an exceeding of the measuring range is 

        Return:
            width clip (float): Returns the length in which the signal is greater than the measuring range. 
        """
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
        return np.max(list_peaks)


def running_mean(x):
    """This function calculates a moving average of a time series of data. Here N is the sample interval over which the smoothing takes place.
        
        Args:
            x (list): Time series to be smoothed

        Returns:
            smoothed data (list): Returns the smoothed data
        """
    N = 20 # über wie viele Werte wird geglättet
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def get_slope(x,t):
    """This function calculates the slope of a peak from exceeding the threshold to the maximum.

        Args:
            x (list): x Values from which the slope is to be determined
            t (list): time section from which the slope is to be determined

        Returns:
            slope (float): slope of the section
        """
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


def evaluate_sensor(df, sensor, threshold):
    """This function calculates the slope of a peak from exceeding the threshold to the maximum.

        Args:
            df (pandas.DataFrame): DateFrame with all sensors from one measurement
            sensor (string): sensor to evaluate
            threshold (float): Value from which an exceeding of the measuring range is determined

        Return:
            peaks (numpy.array): extracted peaks
            properties (dictionary): properties of measurement
            results_half (numpy.array): extracted feauters from peak half
            results_full (numpy.array): extracted feauters from peak full
            result_dict (dictionary): dictionary with extracted feauters
        """
    peaks, peak_properties = find_peaks(df[sensor], prominence=0, width=1, distance=20000, height=threshold)
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

    return (peaks, peak_properties, results_half, results_full, result_dict)


def cut_peakarea(df, sensor_to_cut,sensors_norm):
    """This function cuts out the sigbificant range of a measurement.
    A part from the maximum of the "sensor_to_cut" - "place_before_peak" 
    to the maximum of the "sensor_to_cut" + "place_after_peak" is cut out. 
    In addition, columns with the smoothed data of the corresponding sensors are added. 

    Args:
        df (pandas.DataFrame): DateFrame with all sensors from one measurement
        sensor_to_cut (string): Sensor with which the time period is to be determined
        sensors_norm (list): List of sensors to be normalised
    
    Returns:
        df_corr (pandas.DataFrame): DataFrame with significant range of measurement with smoothed values
    """
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
    df_corr.set_index('time [s]', inplace=True)
    return df_corr

##  saving the result df ##
def save_df(df, path, name):
    """This function saves a DataFrame to csv in the results folder.

        Param:
            df (pandas.DataFrame): DataFrame to save
            path (string): path to root directory of data
            name (string): Name under which the file is to be saved
        """
    path = path + '\\results'
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path + '\\' + name + '.csv'
    print(name + 'saved as ' + path)
    df.to_csv(path, sep=';', decimal=',', index = True)

def read_file(path,decimal,name, path_out, object_raw, properties):
    """This function reads files of the raw data. The data is evaluated
    and features are extracted. A plot is created for each file.
    The function returns a dict with all extracted features

    Args:
        path (string): path to measurements file
        decimal (string): decimal of stored data
        name (string): name of the measurement
        path_out (string): path to save the figures
        object_raw (object): figure object for plotting measurement
        properties (dictionary):  properties from properties json

    Returns:
        dict_result (dictionary): dictionary with all extracted feauters for a measurement
    """
    sensors = properties['sensors']
    path = path + path[path.rfind('\\'):] + '.txt'
    dict_result = {}
    df_measurement = pd.read_csv(path, delimiter='\t', decimal=decimal, dtype=float)
    df_corr = cut_peakarea(df_measurement, properties['sensor_to_cut'], properties['sensors_norm'])
    object_raw.add_item(df_corr, name) # adding data from measurement to df for each sensor including all measurements
    fig = Plot(name,len(df_corr.columns), properties)
    df_corr = df_corr.reindex(sorted(df_corr.columns), axis=1)
    for this_sensor in df_corr.columns:
        peaks, peak_properties, results_half, results_full, this_dict_result = evaluate_sensor(df_corr, this_sensor, sensors[this_sensor]['threshold'])
        dict_result.update(this_dict_result)
        fig.add_subplot(this_sensor, df_corr, peak_properties, results_half, results_full, peaks)
    fig.show_fig(path_out)
    return dict_result



