path = "./ICIAR2018_BACH_Challenge"

import os
import random

new_path = "./Bach_Breakhis/Bach"
data_percent = 0.9

# Move all the tif files in "Benign" and "Normal" folders to subfolder 0 in new path
for folder in os.listdir(path):
    if folder == "Benign":
        for file in os.listdir(path + "/" + folder):
            files = [f for f in os.listdir(path + "/" + folder) if f.endswith(".tif")]
            random_files = random.sample(files, int(len(files) * data_percent))
            # Move  remaining files to val folder
            for file in files:
                if file not in random_files:
                    os.rename(path + "/" + folder + "/" + file, new_path + "/val/0/" + file)

            for file in random_files:
                os.rename(path + "/" + folder + "/" + file, new_path + "/train/0/" + file)
            

    elif folder == "Normal":
        for file in os.listdir(path + "/" + folder):
            files = [f for f in os.listdir(path + "/" + folder) if f.endswith(".tif")]
            random_files = random.sample(files, int(len(files) * data_percent))
            # Move  remaining files to val folder
            for file in files:
                if file not in random_files:
                    os.rename(path + "/" + folder + "/" + file, new_path + "/val/0/" + file)

            for file in random_files:
                os.rename(path + "/" + folder + "/" + file, new_path + "/train/0/" + file)





    elif folder == "InSitu":
        for file in os.listdir(path + "/" + folder):
            files = [f for f in os.listdir(path + "/" + folder) if f.endswith(".tif")]
            random_files = random.sample(files, int(len(files) * data_percent))
            # Move  remaining files to val folder
            for file in files:
                if file not in random_files:
                    os.rename(path + "/" + folder + "/" + file, new_path + "/val/1/" + file)

            for file in random_files:
                os.rename(path + "/" + folder + "/" + file, new_path + "/train/1/" + file)
            


    elif folder == "Invasive":
        for file in os.listdir(path + "/" + folder):
            files = [f for f in os.listdir(path + "/" + folder) if f.endswith(".tif")]
            random_files = random.sample(files, int(len(files) * data_percent))
            # Move  remaining files to val folder
            for file in files:
                if file not in random_files:
                    os.rename(path + "/" + folder + "/" + file, new_path + "/val/1/" + file)
                    
            for file in random_files:
                os.rename(path + "/" + folder + "/" + file, new_path + "/train/1/" + file)



