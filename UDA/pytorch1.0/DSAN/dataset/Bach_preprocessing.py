path = "./ICIAR2018_BACH_Challenge"

import os

new_path = "./Bach_Breakhis/Bach"

# Move all the tif files in "Benign" and "Normal" folders to subfolder 0 in new path
for folder in os.listdir(path):
    if folder == "Benign":
        for file in os.listdir(path + "/" + folder):
            if file.endswith(".tif"):
                os.rename(path + "/" + folder + "/" + file, new_path + "/0/" + file)
    elif folder == "Normal":
        for file in os.listdir(path + "/" + folder):
            if file.endswith(".tif"):
                os.rename(path + "/" + folder + "/" + file, new_path + "/0/" + file)
    elif folder == "InSitu":
        for file in os.listdir(path + "/" + folder):
            if file.endswith(".tif"):
                os.rename(path + "/" + folder + "/" + file, new_path + "/1/" + file)
    elif folder == "Invasive":
        for file in os.listdir(path + "/" + folder):
            if file.endswith(".tif"):
                os.rename(path + "/" + folder + "/" + file, new_path + "/1/" + file)    

