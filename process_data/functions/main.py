"""
This is the main module of this libary. 
It reads the raw data, evaluates them and creates plots.
comment packages out you dont want to run
Set up the path to data in `properties.josn <https://github.com/mattmatt91/Promotion_process/blob/66fb5dd8f33c64ae7a507ba354351cdf820e7932/process_data/functions/properties.json>`_

Find the `main.py <https://github.com/mattmatt91/Promotion_process/blob/66fb5dd8f33c64ae7a507ba354351cdf820e7932/process_data/functions/main.py>`_

"""

from read_files import scan_folder, extract_properties
from do_statistics import calculate
from compare_measurements import compare
from plot_feauters import  plot_features

if __name__ == '__main__':
    
    properties = extract_properties()
    root_path = properties['root_path']

    scan_folder(root_path, properties) # reading raw data
    calculate(root_path, properties, statistic=True, pca=True, lda=True, browser=True, dimension=False) # computing statistics
    compare(root_path, properties) # evaluates the files with one sensor, all measurments
    plot_features(root_path, properties) # plots feauteres with all samples 
    
