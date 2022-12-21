# contains all the constants used in the project

import os
from os.path import dirname
import numpy as np

rot_angle = np.arctan(16/97)*(180/np.pi)
img_rows = 100  # patch height
img_cols = 100  # patch width

num_classes = 17  # number of classes in the data set

osm_features = 56  # number of osm based features

if os.name == "nt":  # locally
    current_dir_path = dirname(dirname((os.getcwd())))

# paths to So2Sat POP Part1 folder
all_patches_mixed_part1 = os.path.join(current_dir_path, 'So2Sat_POP_Part1')  # path to So2Sat POP Part 1 data folder
all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, 'train')   # path to train folder
all_patches_mixed_test_part1 = os.path.join(all_patches_mixed_part1, 'test')   # path to test folder

# paths to So2Sat POP Part2 folder
all_patches_mixed_part2 = os.path.join(current_dir_path, 'So2Sat_POP_Part2')  # path to So2Sat POP Part 2 data folder
all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, 'train')   # path to train folder
all_patches_mixed_test_part2 = os.path.join(all_patches_mixed_part2, 'test')   # path to test folder

# path to configuration folder
src_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(src_path, 'config')
exp_path = os.path.join(os.path.dirname(src_path), 'results')  # path to Experiment Directoy

# path to US cities
us_city_rasters = os.path.join(current_dir_path, 'city_rasters_us')
all_patches_mixed_us_part1 = os.path.join(current_dir_path, 'US_POP_Part1')
all_patches_mixed_us_part2 = os.path.join(current_dir_path, 'US_POP_Part2')

# path to EU cities to compare with GHS-POP
city_rasters = os.path.join(current_dir_path, 'city_rasters/ghs_vs_ours_cls/')

class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8',
               'Class_9', 'Class_10', 'Class_11', 'Class_12', 'Class_13', 'Class_14', 'Class_15', 'Class_16']
