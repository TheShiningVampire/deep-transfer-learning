path = './Breakhis_large'

import os

new_path = './Bach_Breakhis/Breakhis'

# Copy all files in the subfolders of 0 in path to 0 in new_path
for folder in os.listdir(path):
    if folder == "0":
        for sub1 in os.listdir(path + "/" + folder):
            for sub2 in os.listdir(path + "/" + folder + "/" + sub1):
                for sub3 in os.listdir(path + "/" + folder + "/" + sub1 + "/" + sub2):
                    for file in os.listdir(path + "/" + folder + "/" + sub1 + "/" + sub2 + "/" + sub3):
                        os.rename(path + "/" + folder + "/" + sub1 + "/" + sub2 + "/" + sub3 + "/" + file, new_path + "/" + folder + "/" + file)
    elif folder == "1":
        for sub1 in os.listdir(path + "/" + folder):
            for sub2 in os.listdir(path + "/" + folder + "/" + sub1):
                for sub3 in os.listdir(path + "/" + folder + "/" + sub1 + "/" + sub2):
                    for file in os.listdir(path + "/" + folder + "/" + sub1 + "/" + sub2 + "/" + sub3):
                        os.rename(path + "/" + folder + "/" + sub1 + "/" + sub2 + "/" + sub3 + "/" + file, new_path + "/" + folder + "/" + file)



