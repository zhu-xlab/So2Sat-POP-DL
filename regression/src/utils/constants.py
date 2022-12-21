# contains all the constants used in the project

import os
import numpy as np
from os.path import dirname

rot_angle = np.arctan(16/97)*(180/np.pi)
img_rows = 100  # patch height
img_cols = 100  # patch width

osm_features = 56  # number of osm based features
num_classes = 1  # one for regression

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

src_path = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(src_path, 'config')
exp_path = os.path.join(os.path.dirname(src_path), 'results')
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

us_city_rasters = os.path.join(current_dir_path, 'city_rasters_us')
all_patches_mixed_us_part1 = os.path.join(current_dir_path, 'US_POP_Part1')
all_patches_mixed_us_part2 = os.path.join(current_dir_path, 'US_POP_Part2')

# all osm features
osm_feature_names = ['aerialway', 'aeroway', 'amenity', 'barrier', 'boundary', 'building', 'craft', 'emergency', 'geological',
'healthcare', 'highway', 'historic', 'landuse', 'leisure', 'man_made', 'military', 'natural',
'office', 'place', 'power', 'public_transport', 'railway', 'route', 'shop', 'sport', 'telecom',
'tourism', 'water', 'waterway', 'addr:housenumber', 'restrictions', 'other', 'n', 'm', 'k_avg',
'intersection_count', 'streets_per_node_avg', 'streets_per_node_counts_argmin',
'streets_per_node_counts_min', 'streets_per_node_counts_argmax', 'streets_per_node_counts_max',
'streets_per_node_proportions_argmin', 'streets_per_node_proportions_min',
'streets_per_node_proportions_argmax', 'streets_per_node_proportions_max', 'edge_length_total',
'edge_length_avg', 'street_length_total', 'street_length_avg', 'street_segment_count',
'node_density_km', 'intersection_density_km', 'edge_density_km', 'street_density_km', 'circuity_avg',
'self_loop_proportion']

# only imp. features for mapping colors in plots
osm_prominent_feature_names = ["edge_length_total", "intersection_count", "k_avg", "street_length_total",
                               "streets_per_node_counts_argmin", "streets_per_node_counts_max", "streets_per_node_counts_min",
                               "streets_per_node_proportions_max", "streets_per_node_proportions_min"]

dpi = 96
fig_size = (400 / dpi, 400 / dpi)
fig_size_heatmap = (400 / dpi, 400 / dpi)
