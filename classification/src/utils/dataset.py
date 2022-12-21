import os

import pathlib
import rasterio
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F

from joblib import load
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from utils.constants import img_rows, img_cols, osm_features, num_classes, config_path
from utils.file_folder_ops import load_json

mm_scaler = load(os.path.join(config_path, 'dataset_stats', 'mm_scaler.joblib'))


class PopulationDataset(Dataset):
    """
          Pytorch DatasetClass for Population Data So2Sat POP
          -----------
          :Args:
              list_IDs: List containing the path to all files for the modality "spring"
              labels: List containing the Labels (Population Class)
              dim=(img_rows, img_cols): dimension of the images
              n_classes: how many classes there are (for Classification Case)
              transform: (bool) If transformation should be applied
              test: (bool) If we predict on test set
      """
    def __init__(self, list_IDs, labels, dim=(img_rows, img_cols), n_classes=num_classes, transform=None):
        self.dim = dim
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ID_temp = self.list_IDs[idx]

        # Generate data
        X, y = self.data_generation(ID_temp)
        y = F.one_hot(torch.tensor(y), num_classes=self.n_classes)
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = y.float()
        sample = {'input': X, 'label': y}

        return sample

    def data_generation(self, ID_temp):
        # Initialization
        sen2spr_X = np.empty((*self.dim, 3))
        viirs_X = np.empty((*self.dim, 1))
        osm_X = np.empty((osm_features, 1))
        lcz_X = np.empty((*self.dim, 1))
        lu_X = np.empty((*self.dim, 4))
        dem_X = np.empty((*self.dim, 1))

        # preparing the batch from other datasets
        ID_spring = ID_temp  # batch from sen2 autumn
        ID_viirs = ID_temp.replace('sen2spring', 'viirs')
        ID_osm = ID_temp.replace('sen2spring', 'osm_features').replace('tif', 'csv')
        ID_lcz = ID_temp.replace('sen2spring', 'lcz')
        ID_lu = ID_temp.replace('sen2spring', 'lu')
        ID_dem = ID_temp.replace('Part1', 'Part2').replace('sen2spring', 'dem')

        sen2spr_X, y = generate_data(sen2spr_X, ID_spring, channels=3, data='sen2spring')
        viirs_X, y = generate_data(viirs_X, ID_viirs, channels=1, data='viirs')
        osm_X, y = generate_osm_data(osm_X, ID_osm, mm_scaler, channels=1)
        lcz_X, y = generate_data(lcz_X, ID_lcz, channels=1, data='lcz')
        lu_X, y = generate_data(lu_X, ID_lu, channels=4, data='lu')
        dem_X, y = generate_data(dem_X, ID_dem, channels=1, data='dem')
        y = F.one_hot(torch.tensor(y), num_classes=self.n_classes)
        y = y.float()
        return np.concatenate((sen2spr_X, viirs_X, lcz_X, lu_X, dem_X), axis=0), y, osm_X


class PopulationDataset_Clf(PopulationDataset):
    '''
        Population Dataset for Standard Classification Task
    '''

    def __init__(self, list_IDs, labels, dim=(img_rows, img_cols), n_classes=num_classes, transform=None, test=False):
        super(PopulationDataset_Clf, self).__init__(list_IDs, labels, dim, n_classes, transform)
        self.test = test

    def __getitem__(self, idx):
        ID_temp = self.list_IDs[idx]
        # Generate data
        X, y, osm = self.data_generation(ID_temp)
        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)
        ID = ID_temp.split(os.sep)[-1].split('_sen2')[0]
        osm = torch.from_numpy(osm).type(torch.FloatTensor)
        if self.transform:
            X = self.transform(X)

        sample = {'input': X, 'label': y, 'osm': osm, 'identifier': ID}
        return sample


def generate_data(X, ID_temp, channels, data):
    # load dataset statistics and patches
    dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'mod_dataset_stats.json'))
    with rasterio.open(ID_temp, 'r') as ds:
        if 'sen2' in data:
            image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.cubic)
            if image.shape[0] == 13:
                #  for new sentinel-2 images
                image = image[1:4]
                image = image[::-1, :, :]
        elif data == 'lcz':
            image = ds.read(out_shape=(ds.count, img_rows, img_cols))
        else:
            image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)
    new_arr = np.empty([channels, img_rows, img_cols])
    for k, layer in enumerate(image):
        if data == 'lcz':
            arr = np.zeros([1, img_rows, img_cols])
            for cl in range(0, 18):
                if cl == 0:
                    arr[layer.reshape(1, 100, 100) == cl] = 0.1
                elif cl > 0 and cl <= 10:
                    scaler = -0.09 * cl + 1.09
                    arr[layer.reshape(1, 100, 100) == cl] = scaler
                else:
                    arr[layer.reshape(1, 100, 100) >= cl] = 0
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
        else:
            channel_min = dataset_stats[data]['min'][k]
            channel_max = dataset_stats[data]['max'][k]
            arr = layer
            if data == 'viirs':
                p2 = dataset_stats[data]['p2'][k]
                arr = np.where(arr < 0, 0, arr)
                arr = np.where(arr >= p2, p2, arr)
            arr = arr - channel_min
            arr = arr / (channel_max - channel_min)
            new_arr[k] = arr
    X = new_arr
    # Store class
    path = pathlib.PurePath(ID_temp)
    folder_name = path.parent.name
    y = int(folder_name.rsplit('_')[1])

    return X, y


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

    # Store class
    path = pathlib.PurePath(ID_temp)
    folder_name = path.parent.name
    y = int(folder_name.rsplit('_')[1])

    return X, y
