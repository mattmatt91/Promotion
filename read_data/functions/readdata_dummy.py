"""
File:                       USB1808_a_in_scan_continuous.py

Library Call Demonstrated:  mcculw.ul.a_in_scan() in Background mode with scan
                            options BACKGROUND, CONTINUOUS and, SCALEDATA.

Purpose:                    Scans a range of A/D Input Channels and stores
                            the sample data in an array.

Demonstration:              Displays the analog input from two channels
                            during a continuously running scan.

Other Library Calls:        mcculw.ul.get_board_name()
                            mcculw.ul.flash_LED()
                            mcculw.ul.a_chan_input_mode()
                            mcculw.ul.scaled_win_buf_alloc()
                            mcculw.ul.win_buf_free()
                            mcculw.ul.get_status()
                            mcculw.ul.stop_background()

Special Requirements:       Analog signals on up to two input channels.
                            For example, for single ended, connect terminals:
                            CH0H to AGND,
                            CH1H to +VO.

Notes:                      No board detection or device discovery in this app.
                            Device must be assigned in InstaCal as Board 0.
                            This example is made simple so as not to
                            be confusing (that was the hope anyway).
"""
from __future__ import absolute_import, division, print_function
from builtins import *  # @UnusedWildImport

from time import sleep, time_ns, time
from ctypes import cast, POINTER, c_double

from mcculw import ul
from mcculw.enums import ULRange, AnalogInputMode, ScanOptions, \
    FunctionType, Status, DigitalPortType,  DigitalIODirection
from mcculw.device_info import DaqDeviceInfo

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randrange

import random

def read_data(properties, path):
    # print(properties)
    channel_num = 0
    device_to_show = "USB-1808"
    board_num = 0
    rate = properties['rate']
    points_per_channel = 10000
    memory_handle = None
    ai_range = ULRange.BIP10VOLTS
    max_samples = rate*properties['duration']
    droptime = properties['droptime']
    low_channel = 0
    high_channel = len(properties['sensors'])-1
    num_channels = high_channel - low_channel + 1
    data = []
    for i in range(max_samples):
        this_samples = (i%200000)
        this_data = [np.sin(this_samples), np.sin(this_samples//2), np.sin(this_samples*2), np.tan(this_samples), np.tan(this_samples//2), np.tan(this_samples*2)]
        this_data = [i*random.uniform(0, 1) for i in this_data]
        data.append(this_data)
    #data = np.full([max_samples, num_channels], 1)
    df = pd.DataFrame(data, columns=properties['sensors'])
    df['time [s]'] = [i/rate for i in df.index]
    df.set_index('time [s]', inplace=True) 
    print(df.head())
    print(df.info())
    df.to_csv(path, sep='\t', decimal='.', index=True)
    # df.plot()
    # plt.show()
        
        



