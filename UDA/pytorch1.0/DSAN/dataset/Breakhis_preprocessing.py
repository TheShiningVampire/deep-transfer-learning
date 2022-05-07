path = "./BreaKHis 400X"

import os

new_path = "./Bach_Breakhis/Breakhis"

# For both train and test folder, move files in benign to 0 and malignant to 1
for folder in os.listdir(path):
    if folder == "train":
        for folder_ in os.listdir(path + "/" + folder):
            if folder_ == "benign":
                for file in os.listdir(path + "/" + folder + "/" + folder_):
                    os.rename(path + "/" + folder + "/" + folder_ + "/" + file, new_path + "/0/" + file)
            elif folder_ == "malignant":
                for file in os.listdir(path + "/" + folder + "/" + folder_):
                    os.rename(path + "/" + folder + "/" + folder_ + "/" + file, new_path + "/1/" + file)
    elif folder == "test":
        for folder_ in os.listdir(path + "/" + folder):
            if folder_ == "benign":
                for file in os.listdir(path + "/" + folder + "/" + folder_):
                    os.rename(path + "/" + folder + "/" + folder_ + "/" + file, new_path + "/0/" + file)
            elif folder_ == "malignant":
                for file in os.listdir(path + "/" + folder + "/" + folder_):
                    os.rename(path + "/" + folder + "/" + folder_ + "/" + file, new_path + "/1/" + file)

