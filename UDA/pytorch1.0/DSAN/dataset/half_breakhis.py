path = "./Bach_Breakhis/Breakhis_large/1"
target_path = "./Bach_Breakhis/Breakhis/1"
data_percent = 0.5

import os
import random

files = [f for f in os.listdir(path) if f.endswith(".png")]

random_files = random.sample(files, int(len(files) * data_percent))

# Move random_files to target_path
for file in random_files:
    os.rename(path + "/" + file, target_path + "/" + file)
