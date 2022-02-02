from datetime import datetime
import json as js
import os
from pathlib import Path

from readdata_dummy_data import read_data 

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]  

def read_json(filename):
    with open(filename) as json_file:
        return js.load(json_file)
    
def get_path(properties):
    path = properties['path']
    last_dir = properties['lastdir']
    print('current directory: {0}'.format(path))
    print('select path')
    print('press:')
    folders = get_immediate_subdirectories(path)
    folders_dict = {1: last_dir}
    print('1 for last dir: ', last_dir)
    i = 2
    for folder in folders:
        folders_dict[i] = folder
        
        print(i, 'for ', folder)
        i += 1   
    print('press 0 for new folder') 
    folder = None
    while folder == None:
        user_input = input()
        try:
            folder = folders_dict[int(user_input)]
        except:
            try:
                if int(user_input) == 0:
                    user_input = input('set foldername: ')
                    directory = str(user_input)
                    this_path = os.path.join(path, directory)
                    Path(this_path).mkdir(parents=True, exist_ok=True)
                    print("Directory '%s' created" %directory)
                    folder = directory
            except:
                pass
                print('folder out of range')
    print(folder, ' selected')
    return folder, path

def select_sample(samples):
    print('select sample')
    print('press:')
    for sample in samples:
        print(sample, 'for', samples[sample])
    sample = None
    while sample == None:
        user_input = input()
        try:
            sample = samples[user_input]
        except:
            print('samlple out of range')
    print(sample, ' selected')
    return sample

def select_height():
    print('select height')
    print('enter int in cm:')
    height = None
    while height == None:
        user_input = input()
        try:
            height = int(user_input)
        except:
            pass
    print('height: ', height, 'cm')
    return height

def set_number():
    print('select number')
    print('enter int:')
    sample_number = None
    while sample_number == None:
        user_input = input()
        try:
            sample_number = int(user_input)
        except:
            print('not an integer')
    print('number: ', sample_number)
    return sample_number

def add_info():
    user_input = input('add info')
    return user_input

def update_properties(properties):
    update_json = {}
    for item in "rate duration sensors path lastdir droptime".split():
        update_json[item]= properties[item]

    json_string = js.dumps(update_json)
    with open('properties.json', 'w') as outfile:
        outfile.write(json_string)
    
def mkdir(properties, folder):
    name = properties['sample'] + '_' + str(properties['sample_number']) + '_' + str(properties['datetime'])
    path = os.path.join(properties['path'], folder, name)
    Path(path).mkdir(parents=True, exist_ok=True)
    path_file = os.path.join(path, name)
    path_file = path_file + '.txt'
    return path, path_file

def save_properties_measurement(properties):
    save_properties = properties
    del save_properties['lastdir']
    json_string = js.dumps(save_properties)
    json_name = properties['path'] + '\\info.json'
    with open(json_name, 'w') as outfile:
        outfile.write(json_string)

def start_measurement(number_of_measurement):
    # reading sample list
    samples = read_json('samples.json')

    # reading properties
    properties = read_json('properties.json')
    # select path
    folder, path = get_path(properties)

    for i in range(number_of_measurement):
        # reading properties
        properties = read_json('properties.json')
        # select path
        properties['lastdir'] = folder
        #select sample
        sample = samples[str((i%5+1))]
        print('sample is {0}'.format(sample))
        # sample = select_sample(samples)
        properties["sample"]= sample

        #select height
        height = 50
        print('height is {0}'.format(height))
        #height = select_height()
        properties["height"]= height

        #select number of sample
        sample_number = i
        print('number is {0}'.format(sample_number))
        #sample_number = set_number()
        properties['sample_number'] = sample_number

        #add info
        info = "this is measurement number {0}".format(i)
        # info = add_info()
        properties['info'] = info
        
        #setting timestamp
        properties["datetime"]= datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        # updating poperties global json
        update_properties(properties)
        
        # make dir for measurement
        path, path_file = mkdir(properties, folder)
        properties['path'] = path

        # saving properties for measurement
        save_properties_measurement(properties)

        # main measuring part
        read_data(properties, path_file)
    
if __name__ == '__main__':
    start_measurement(50)