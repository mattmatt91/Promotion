from read_files import scan_folder, extract_properties
from do_statistics import calculate
from compare_measurements import compare
from plot_feauters import  plot_features

if __name__ == '__main__':
    root_path = "C:\\Users\\mmuhr-adm\\Desktop\\Neuer ZIP-komprimierter Ordner\\Messaufbau\\dataaquisition\\data\\test4"
    properties = extract_properties()

    scan_folder(root_path, properties)
    # calculate(root_path, properties, statistic=True, pca=True, lda=True, svm=False, knn=False, browser=True, dimension=False)
    # compare(root_path, properties) # evaluates the files with one sensor, all measurments
    # plot_features(root_path, properties)
