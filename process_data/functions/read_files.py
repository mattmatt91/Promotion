from os import listdir, scandir, sep
from os.path import isfile, join
from sensors import read_file
from sensors import Sensor
import pandas as pd
from pathlib import Path
import json as js


def read_json(filename):
    with open(filename) as json_file:
        return js.load(json_file)

def extract_info(path):
    dict={}
    path = path + '\\info.json'
    dict = read_json(path)
    return dict

def extract_properties():
    path = str(Path().absolute()) + '\\properties.json'
    print(path)
    dict = read_json(path)
    return dict

def scan_folder(path, properties):
    df_result = pd.DataFrame()
    # properties = extract_properties()
    df_result_raw = Sensor(properties) # dataframe for each sensor with all measurements
    
    # creates list with subfolers
    subfolders = [f.path for f in scandir(path) if f.is_dir()]
    for folder in subfolders:
        if folder.find('\\Results') < 0 and folder.find('\\Bilder', ) < 0 and folder.find('\\results') < 0:
            dict = extract_info(folder)
            name = dict['path'][dict['path'].rfind('\\')+1:dict['path'].rfind('.')]
            print(name)
            dict.update({"name": name})
            dict.update(read_file(folder, '.', name, path, df_result_raw, properties)) #evaluating file
            # Platzhalter # dict.update(analyze_spectra(path_dict['Spektrometer'], path_dict['Spektrometerref'], path, dict['name']))
            df_result = df_result.append(dict, ignore_index=True) # append measurement in result file
    result_path = path + '\\' + 'Results'
    Path(result_path).mkdir(parents=True, exist_ok=True)
    result_path = result_path + '\\Result.csv'
    df_result.to_csv(result_path, decimal=',', sep=';', index = False) # safe the result df
    df_result_raw.save_items(path) # save the sensor df 


# scan_folder("C:\\Users\\Matthias\\Desktop\\Messaufbau\\dataaquisition\\data\\test")