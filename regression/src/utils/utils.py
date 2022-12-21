#  Utility Functions to prepare the data set

import glob
import os

import numpy as np
import pandas as pd


def get_fnames_labs(path):
    """
    Gets Files Names of all files of sen2spring
    :param path: path to patch folder (sen2spring)
    :return: gives the paths of all the tifs and its corresponding class labels
    """
    city_folders = glob.glob(os.path.join(path, "*"))
    f_names_all = np.array([])
    labs_all = np.array([])
    for each_city in city_folders:
        data_folders = glob.glob(os.path.join(each_city, "*"))
        for each_folder in data_folders:
            if each_folder.__contains__('sen2spring'):
                class_folders = glob.glob(os.path.join(each_folder, "*"))
                for folder in class_folders:
                    # get all filenames of *_sen2_rgb_spring.tif
                    f_names = np.array([x for x in glob.glob(os.path.join(folder, "*.tif"))])
                    labs = np.full(len(f_names), os.path.basename(folder))
                    f_names_all = np.append(f_names_all, f_names, axis=0)
                    labs_all = np.append(labs_all, labs, axis=0)
    return f_names_all, labs_all


def get_fnames_labs_reg(path):
    """
    :param path: path to patch folder (sen2spring)
    :return: gives the paths of all the tifs and its corresponding class labels
    """
    city_folders = glob.glob(os.path.join(path, "*"))
    f_names_all = np.array([])
    labs_all = np.array([])
    for each_city in city_folders:
        data_path = os.path.join(each_city, "sen2spring")
        csv_path = os.path.join(each_city, each_city.split(os.sep)[-1:][0] + '.csv')
        city_df = pd.read_csv(csv_path)
        ids = city_df['GRD_ID']
        pop = city_df['POP']
        classes = city_df['Class']
        classes_str = [str(x) for x in classes]
        classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
        for index in range(0, len(classes_paths)):
            f_names = [classes_paths[index] + str(ids[index]) + '_sen2spring.tif']
            f_names_all = np.append(f_names_all, f_names, axis=0)
            labs = [pop[index]]
            labs_all = np.append(labs_all, labs, axis=0)

    return f_names_all, labs_all


def get_cities(path):
    return glob.glob(os.path.join(path, "*"))


def get_fnames_labs_citywise_reg(city):
    """
    :param path: path to patch folder (sen2spring)
    :return: gives the paths of all the tifs and its corresponding class labels
    """
    f_names_all = np.array([])
    labs_all = np.array([])
    data_path = os.path.join(city, "sen2spring")
    csv_path = os.path.join(city, city.split(os.sep)[-1:][0] + '.csv')
    city_df = pd.read_csv(csv_path)
    ids = city_df['GRD_ID']
    pop = city_df['POP']
    classes = city_df['Class']
    classes_str = [str(x) for x in classes]
    classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
    for index in range(0, len(classes_paths)):
        f_names = [classes_paths[index] + ids[index] + '_sen2spring.tif']
        f_names_all = np.append(f_names_all, f_names, axis=0)
        labs = [pop[index]]
        labs_all = np.append(labs_all, labs, axis=0)

    return f_names_all, labs_all
