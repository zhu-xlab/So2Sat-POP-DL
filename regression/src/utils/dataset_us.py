import json
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from joblib import load
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from utils.constants import img_rows, img_cols, osm_features, num_classes, config_path

mm_scaler = load(os.path.join(config_path, 'dataset_stats', 'mm_scaler_us.joblib'))


def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class PopulationDataset(Dataset):
    """
        Pytorch DatasetClass for Population Data So2Sat POP
        -----------
        :Args:
            list_IDs: List containing the path to all files for the modality "autumn"
            labels: List containing the Labels (Population Count)
            dim=(img_rows, img_cols): dimension of the images
            n_classes: how many classes there are (for Classification Case)
            transform: (bool) If transformation should be applied
            test: (bool) If we predict on test set
    """

    def __init__(self, list_IDs, labels, dim=(img_rows, img_cols), n_classes=num_classes, transform=None, test=False,
                 mode=None):
        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.transform = transform
        self.test = test
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def data_generation(self, ID_temp):
        # Initialization
        if ID_temp is None:
            sen2spr_X = np.zeros((3, *self.dim))
            viirs_X = np.zeros((1, *self.dim))
            osm_X = np.zeros((osm_features, 1))
            lcz_X = np.zeros((1, *self.dim))
            lu_X = np.zeros((4, *self.dim))
            dem_X = np.zeros((1, *self.dim))
        else:
            sen2spr_X = np.empty((*self.dim, 3))
            viirs_X = np.empty((*self.dim, 1))
            osm_X = np.empty((osm_features, 1))
            lcz_X = np.empty((*self.dim, 1))
            lu_X = np.empty((*self.dim, 4))
            dem_X = np.empty((*self.dim, 1))

            ID_spring = ID_temp
            ID_viirs = ID_temp.replace('sen2spring', 'viirs')
            ID_osm = ID_temp.replace('sen2spring', 'osm_features').replace('tif', 'csv')
            ID_lcz = ID_temp.replace('sen2spring', 'lcz')
            ID_lu = ID_temp.replace('sen2spring', 'lu')
            ID_dem = ID_temp.replace('Part1', 'Part2').replace('sen2spring', 'dem')

            sen2spr_X = generate_data(sen2spr_X, ID_spring, channels=3, data='sen2spring')
            viirs_X = generate_data(viirs_X, ID_viirs, channels=1, data='viirs')
            osm_X = generate_osm_data(osm_X, ID_osm, mm_scaler, channels=1)
            lcz_X = generate_data(lcz_X, ID_lcz, channels=1, data='lcz')
            lu_X = generate_data(lu_X, ID_lu, channels=4, data='lu')
            dem_X = generate_data(dem_X, ID_dem, channels=1, data='dem')

            return np.concatenate((sen2spr_X, viirs_X, lcz_X, lu_X, dem_X), axis=0), osm_X


class PopulationDataset_Reg(PopulationDataset):
    '''
    Population Dataset for Standard Regression Task
    '''

    def __init__(self, list_IDs, labels, dim=(img_rows, img_cols), transform=None, test=False, mode=None):
        super(PopulationDataset_Reg, self).__init__(list_IDs=list_IDs,
                                                    labels=labels,
                                                    dim=dim,
                                                    transform=transform,
                                                    test=test,
                                                    mode=mode)

    def __getitem__(self, idx):
        ID_temp = self.list_IDs[idx]
        # Generate data
        X, osm = self.data_generation(ID_temp)
        ID = ID_temp
        y = self.labels[idx]
        if self.test == False:
            y = normalize_reg_labels(y)
        else:
            y = y
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)
        osm = torch.from_numpy(osm).type(torch.FloatTensor)
        if self.transform:
            X = self.transform(X)
        sample = {'input': X, 'label': y, 'osm': osm, 'identifier': ID}
        return sample


def normalize_reg_labels(y):
    y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats_us.json'))
    y_max = np.float(y_stats['max'])
    y_min = np.float(y_stats['min'])
    y_scaled = (y - y_min) / (y_max - y_min)
    return y_scaled


def denormalize_reg_labels(y_scaled):
    y_stats = load_json(os.path.join(config_path, 'dataset_stats', 'label_stats_us.json'))
    y_max = np.float(y_stats['max'])
    y_min = np.float(y_stats['min'])
    y = y_scaled * (y_max - y_min) + y_min
    return y


def generate_data(X, ID_temp, channels, data):
    # load dataset statistics and patches
    dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'mod_dataset_stats.json'))
    with rasterio.open(ID_temp, 'r') as ds:
        if 'sen2' in data:
            image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.cubic)
            if image.shape[0] == 13:
                ## for new sentinel-2 images
                image = image[1:4]
                image = image[::-1, :, :]
        elif data == 'lcz':
            image = ds.read(out_shape=(ds.count, img_rows, img_cols))
        else:
            image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)
    new_arr = np.empty([channels, img_rows, img_cols])
    for k, layer in enumerate(image):
        if data == 'lcz':
            arr = layer
            arr = np.where((arr > 0) & (arr <= 10), arr * (-0.09) + 1.09, arr)
            arr = np.where(arr == 0, 0.1, arr)
            arr = np.where(arr > 10, 0, arr)
            new_arr[k] = arr
        elif 'sen2' in data:
            p2 = dataset_stats[data]['p2'][k]
            mean = dataset_stats[data]['mean'][k]
            std = dataset_stats[data]['std'][k]
            arr = layer
            arr = np.where(arr >= p2, p2, arr)
            arr = arr - mean
            arr = arr / std
            new_arr[k] = arr
        elif data == 'viirs':
            p2 = dataset_stats[data]['p2'][k]
            mean = dataset_stats[data]['mean'][k]
            std = dataset_stats[data]['std'][k]
            arr = layer
            arr = np.where(arr < 0, 0, arr)
            arr = np.where(arr >= p2, p2, arr)
            arr = arr - mean
            arr = arr / std
            new_arr[k] = arr
        else:
            channel_min = dataset_stats[data]['min'][k]
            channel_max = dataset_stats[data]['max'][k]
            arr = layer
            arr = arr - channel_min
            arr = arr / (channel_max - channel_min)
            new_arr[k] = arr
    X = new_arr

    return X


def generate_osm_data(X, ID_temp, mm_scaler, channels):
    # Generate data
    # load csv
    df = pd.read_csv(ID_temp, header=None)[1]
    df = df[df.notna()]

    df_array = np.array(df)
    df_array[df_array == np.inf] = 0

    new_arr = np.empty([channels, osm_features])
    new_arr[0] = df_array

    # fit and transform the data
    new_arr = mm_scaler.transform(new_arr)
    scaled_arr = np.empty([channels, osm_features])
    scaled_arr[0] = new_arr
    X = np.transpose(scaled_arr, (1, 0))

    return X
