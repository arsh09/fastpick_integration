import os
import shutil

directory='/home/rick_robofruit/localdrive/robofruit_weightdataset/dataset'

new_directory = '/home/rick_robofruit/localdrive/robofruit_weightdataset/newdataset'

# Access the directory path, name and files
for dirpath, dirnames, files in os.walk(directory):
    # Name of files
    basename = os.path.basename(dirpath)
    for file_name in files:
        print(file_name)
        if file_name.endswith('dump.npy'):
            print(file_name)
            # Location of file on PC
            opening = os.path.join(dirpath, file_name)
            # Sub folder in new directory
            foldern = os.path.join(new_directory, basename)
            # Create sub folder if it does not exist
            if os.path.isdir(foldern) is False:
                makef = os.makedirs(foldern)

            # Create container in new directory
            filen = os.path.join(foldern, file_name)
            # Copy into that file
            shutil.copyfile(opening, filen)

            print(file_name)