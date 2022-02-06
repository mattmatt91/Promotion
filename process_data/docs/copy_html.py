from re import T
import shutil
import os


def copy_files():
    current_directory = os.getcwd()
    print(current_directory)
    source_directory = os.path.join(current_directory, "build", "html")
    print(source_directory)
    destination_directory = os.path.dirname(current_directory)
    destination_directory = os.path.join(os.path.dirname(destination_directory), "docs")
    print(destination_directory)
    try:
        shutil.rmtree(destination_directory)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
    shutil.copytree(source_directory, destination_directory)


if __name__ == '__main__':
    copy_files()