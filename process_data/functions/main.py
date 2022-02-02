from read_files import scan_folder, extract_properties
from statistics import calculate
from compare_measurements import compare
from plot_feauters import  plot_features

if __name__ == '__main__':
    root_path = "C:\\Users\\Matthias\\Desktop\\Messaufbau\\dataaquisition\\data\\test_auto"
    properties = extract_properties()

    # scan_folder(root_path, properties)
    calculate(root_path, properties, statistic=True, pca=True, lda=True, svm=False, knn=False, browser=True, dimension=False)
    # compare(root_path, properties) # evaluates the files with one sensor, all measurments
    #plot_param(root_path, properties)
    # test