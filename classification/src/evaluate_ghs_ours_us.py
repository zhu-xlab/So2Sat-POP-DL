# compares our predictions with ghs-pop

import glob

import os
import math
from collections import OrderedDict

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely import wkt
from sklearn.metrics import precision_score, f1_score, recall_score, balanced_accuracy_score, accuracy_score
from torch.utils.data import DataLoader
from utils.constants import current_dir_path, exp_path, all_patches_mixed_us_part1, img_rows, img_cols
from src.utils.dataset_us import PopulationDataset_Clf
from utils.file_folder_ops import save_json
from utils.utils import get_cities
from models.classification import EO2ResNet_OSM


def define_pop_class(pop):
    # assign population class based on pop account
    min_power = 0
    max_power = 1
    pop_count = int(pop)
    pop_assigned = True
    if pop_count == -340282346638528859811704183484516925440 or pop_count == -200:
        pop_class = 'nan'
        pop_assigned = False

    if pop_count == 0:
        # print("zero pop class found")
        pop_class = 0
        pop_assigned = False

    while pop_assigned:
        if pow(2, min_power) <= pop_count < pow(2, max_power):
            pop_class = max_power
            pop_assigned = False
        else:
            min_power += 1
            max_power += 1
    return pop_class


def plot_by_pop(pop_shp_path, plot_path, csv_path, title, column_name):
    # plots the pop as per the cells of the shapefile in the city
    pop_shp = fiona.open(pop_shp_path)
    crs = pop_shp.crs
    df = pd.read_csv(csv_path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs=crs)
    fig, ax = plt.subplots(1, 1)
    ax.set(title=title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    gdf.plot(column=column_name, ax=ax, legend=True, cax=cax, cmap='jet', vmin=0, vmax=16)
    ax.set_axis_off()
    plt.savefig(plot_path)


def evaluate_ghspop_ours(pred_csv_path):
    """
    evaluates the meterics, compares with GHS-POP and
    Args:
        path to our predictions csv file
    """
    city_rasters = os.path.join(current_dir_path, 'city_rasters/ghs_vs_ours_us/')
    all_cities = glob.glob(os.path.join(city_rasters, '*'))
    for city_path in all_cities:
        gt_list = []
        pr_list = []
        ghs_list = []
        i_list = []
        j_list = []
        current_city_rasters = os.path.join(city_rasters, city_path.rsplit('\\')[-1:][0])
        eu_raster_path = glob.glob(os.path.join(current_city_rasters, '*uspop_count.tif'))[0]
        ghs_raster_path = glob.glob(os.path.join(current_city_rasters, '*ghs_pop_count.tif'))[0]
        log_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_cls_log_us.txt')

        # opens ghs pop count tif as array and convert pop count to pop class
        ghs_src = rasterio.open(ghs_raster_path)
        ghs_raster = ghs_src.read(1)
        ghs_pop_class = np.zeros([ghs_raster.shape[0], ghs_raster.shape[1]])
        for row, elements in enumerate(ghs_raster):
            for i in range(len(elements)):
                pop_class = define_pop_class(int(elements[i]))
                ghs_pop_class[row][i] = pop_class

        # opens reference count tif as array and convert pop count to pop class
        eu_src = rasterio.open(eu_raster_path)
        eu_raster = eu_src.read(1)
        eu_pop_class = np.zeros([eu_raster.shape[0], eu_raster.shape[1]])
        for row, elements in enumerate(eu_raster):
            for i in range(len(elements)):
                pop_class = define_pop_class(int(elements[i]))
                eu_pop_class[row][i] = pop_class

        # excludes the nan values
        for i in range(0, eu_pop_class.shape[0]):
            eu_row = eu_pop_class[i]
            ghs_row = ghs_pop_class[i]
            for j in range(len(eu_row)):
                if not math.isnan(eu_row[j]):
                    gt_list.append(eu_row[j])
                    if math.isnan(ghs_row[j]):
                        gt_list.remove(eu_row[j])
                        i_list.append(i)
                        j_list.append(j)
                        eu_pop_class[i][j] = 'nan'
                    else:
                        ghs_list.append(ghs_row[j])
                else:
                    ghs_pop_class[i][j] = 'nan'

        # calculate the evaluation metrics on ghspop pop estimates
        val_f1 = round(f1_score(gt_list, ghs_list, average='macro'), 4)
        val_recall = round(recall_score(gt_list, ghs_list, average='macro'), 4)
        val_precis = round(precision_score(gt_list, ghs_list, average='macro'), 4)
        val_macd = np.mean(np.abs(np.array(gt_list) - np.array(ghs_list)))
        accuracy = accuracy_score(gt_list, ghs_list)
        balanced_accuracy = balanced_accuracy_score(gt_list, ghs_list)
        with open(log_path, 'w') as f:
            f.writelines(
                "\n \n GHS POP Results: \n — test_f1_macro: {} \n  — test_precis_macro: {} \n — test_recall_macro: {} "
                "\n — test_macd: {} \n — test_accuracy: {} \n — test_balanced_accuracy: {} \n".format(val_f1,
                                                                                                      val_precis,
                                                                                                      val_recall,
                                                                                                      val_macd,
                                                                                                      accuracy,
                                                                                                      balanced_accuracy))

        ghs_plot_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_cls_ghs_us.png')
        fig, ax = plt.subplots(1, 1)
        title = 'GHS-POP population by class'

        im = ax.imshow(ghs_pop_class, cmap='jet', vmin=0, vmax=16)
        ax.set(title=title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_axis_off()
        plt.savefig(ghs_plot_path)

        # plots the reference pop class
        gt_plot_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_cls_gt_us.png')
        fig, ax = plt.subplots(1, 1)
        title = 'Target population by class'

        im = ax.imshow(eu_pop_class, cmap='jet', vmin=0, vmax=16)
        ax.set(title=title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation="vertical")
        ax.set_axis_off()
        plt.savefig(gt_plot_path)

        # calculate the evaulation metric on our predictions and plot the pop distribution
        pop_shp_path = glob.glob(os.path.join(current_city_rasters, '*1km.shp'))[0]
        df_pop = gpd.read_file(pop_shp_path)
        df = pd.read_csv(pred_csv_path)
        #df = df[df.GT_POP_CLASS != 0]
        id_list = list(df['GRD_ID'])
        ids = [str(x).rsplit('/')[-1:][0].rsplit('_')[0] for x in id_list]
        df['GRD_ID'] = ids
        gt_pop_list = df['GT_POP_CLASS']
        pr_pop_list = df['PR_POP_CLASS']
        df_sorted = pd.DataFrame()
        for i in range(0, len(df_pop.id)):
            id = str(int(df_pop.id[i]))
            geometry = df_pop.geometry[i]
            row = df[df.GRD_ID == id]
            row['geometry'] = geometry
            df_sorted = pd.concat([df_sorted, row], ignore_index=True)

        val_f1 = round(f1_score(df_sorted['GT_POP_CLASS'].to_list(), df_sorted['PR_POP_CLASS'].to_list(),
                                average='macro'), 4)
        val_recall = round(recall_score(df_sorted['GT_POP_CLASS'].to_list(), df_sorted['PR_POP_CLASS'].to_list(),
                                        average='macro'), 4)
        val_precis = round(precision_score(df_sorted['GT_POP_CLASS'].to_list(), df_sorted['PR_POP_CLASS'].to_list(),
                                           average='macro'), 4)
        val_macd = np.mean(np.abs(np.array(df_sorted['GT_POP_CLASS'])) - np.array(df_sorted['PR_POP_CLASS']))
        accuracy = accuracy_score(df_sorted['GT_POP_CLASS'].to_list(), df_sorted['PR_POP_CLASS'].to_list())
        balanced_accuracy = balanced_accuracy_score(df_sorted['GT_POP_CLASS'].to_list(),
                                                    df_sorted['PR_POP_CLASS'].to_list())

        with open(log_path, 'a') as f:
            f.writelines(
                "\n\n Our results: \n — test_f1_macro: {} \n  — test_precis_macro: {} \n — test_recall_macro: {} "
                "\n — test_macd: {} \n — test_accuracy: {} \n — test_balanced_accuracy: {} \n".format(val_f1,
                                                                                                      val_precis,
                                                                                                      val_recall,
                                                                                                      val_macd,
                                                                                                      accuracy,
                                                                                                      balanced_accuracy)
            )

        csv_local_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_cls_evaluation_local.csv')
        df_sorted.to_csv(csv_local_path, index=False)
        pr_plot_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_cls_pr_plot_us.png')
        plot_by_pop(pop_shp_path, pr_plot_path, csv_local_path, title='Predicted population by class',
                    column_name='PR_POP_CLASS')
        print('Logs saved at: ', log_path)


def get_fnames_labs_us(path):
    """

    :param path: path to patch folder (sen2autumn)
    :return: gives the paths of all the tifs and its corresponding class labels
    """

    f_names_all = np.array([])
    labs_all = np.array([])

    data_path = os.path.join(path, "sen2spring")
    csv_path = os.path.join(path, path.split(os.sep)[-1:][0] + '.csv')
    city_df = pd.read_csv(csv_path)
    ids = city_df['GRD_ID']
    pop = city_df['POP']
    classes = city_df['Class']
    classes_int = [int(x) for x in classes]
    classes_str = [str(x) for x in classes_int]
    classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
    for index in range(0, len(classes_paths)):
        f_names = [classes_paths[index] + str(ids[index]) + '_sen2spring.tif']
        f_names_all = np.append(f_names_all, f_names, axis=0)
        # for classification
        labs = np.full(len(f_names), (classes_paths[index].rsplit('Class_')[-1]).rsplit('/')[0].rsplit('\\')[0])
        labs_all = np.append(labs_all, labs, axis=0)

    return f_names_all, labs_all


def predict_us_cities(model, model_name, exp_dir, osm_flag, data_dir):
    """
    Predicts on the US cities
    Args:
        model= trained model to be used
        model_name = name of the trained model
        exp_dir = path to result directory
        osm_flag = flsg to use OSM data or not
        data_dir = path to the data directory
    Returns the path to prediction csv
    """
    dataset_name = 'us'
    params = {'dim': (img_rows, img_cols)}
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title = model_name
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name + '.pth'),
                            map_location=torch.device(map_location))
    model_scale = model_dict['hyperparams']['model_scale']
    num_class = model_dict['hyperparams']['num_classes']
    if osm_flag:
        if 'viirs' in model_name:
            print('Evaluation without VIIRS')
            model = model(input_channels=10, num_classes=num_class, scale_factor=model_scale)
        else:
            model = model(input_channels=10, num_classes=num_class, scale_factor=model_scale)
    else:
        model = model(input_channels=10, num_classes=17)
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])
    ## manually loading
    cities = get_cities(data_dir)

    for city in cities:
        print(city)
        ID_list = []
        all_targets = torch.empty(0)
        all_preds = torch.empty(0)
        denormalized_dict = OrderedDict()
        denormalized_dict['preds'] = {}
        denormalized_dict['targets'] = {}
        city_name = os.path.basename(city)
        denormalized_dict['preds'][city_name] = {}
        denormalized_dict['targets'][city_name] = {}
        f_names_test, labels_test = get_fnames_labs_us(city)
        test_dataset = PopulationDataset_Clf(f_names_test, labels_test, test=True, **params)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)
        city_preds = torch.zeros((len(test_dataset)))
        city_targets = torch.zeros((len(test_dataset)))
        for i, data in enumerate(test_loader):
            model.eval()
            if osm_flag:
                inputs, targets, osm, ID = data['input'].to(device), \
                                           data['label'].to(device), data['osm'].to(device), data['identifier'][0]
                with torch.no_grad():
                    preds = model(inputs, osm)
            else:
                inputs, targets, ID = data['input'].to(device), data['label'].to(device), data['identifier'][0]
                with torch.no_grad():
                    preds = model(inputs)
            # preds = preds.view(-1)
            _, predicted = preds.max(1)
            _, labels = targets.max(1)
            city_preds[i * batch_size:i * batch_size + preds.shape[0]] = predicted
            city_targets[i * batch_size:i * batch_size + preds.shape[0]] = labels
            ID_list.append(ID)
            denormalized_dict['preds'][city_name][ID] = predicted.item()
            denormalized_dict['targets'][city_name][ID] = labels.item()  # .cpu().numpy()#.detach().numpy()
        all_targets = torch.cat([all_targets, city_targets])
        all_preds = torch.cat([all_preds, city_preds])
        if not os.path.exists(os.path.join(exp_dir, 'log', title)):
            os.makedirs(os.path.join(exp_dir, 'log', title))
        if not os.path.exists(os.path.join(exp_dir, 'log', title, city_name)):
            os.mkdir(os.path.join(exp_dir, 'log', title, city_name))
        save_json(denormalized_dict,
                  os.path.join(exp_dir, 'log', title, city_name, title + '_cls_allpredictions_id_us.json'))
        df = pd.DataFrame()
        df['GRD_ID'] = ID_list
        df['PR_POP_CLASS'] = all_preds.tolist()
        df['GT_POP_CLASS'] = all_targets.tolist()
        pred_csv = os.path.join(exp_dir, 'log', title, city_name, title + '_evaluation_final_us.csv')
        df.to_csv(pred_csv, index=False)
        return pred_csv


if __name__ == '__main__':
    model_name = 'cl_best_model'  # name of the model to use
    # saves our prediction using the above mentioned model
    pred_csv = predict_us_cities(model=EO2ResNet_OSM, model_name=model_name, exp_dir=exp_path, osm_flag=True,
                      data_dir=all_patches_mixed_us_part1)

    # evalautes and compare the results
    #pred_csv = 'D:/LU_Rasters/Raster_data/dl_popest_so2sat/classification/results/log/cl_best_model/00275_23131_sanjose/cl_best_model_evaluation_final.csv'
    evaluate_ghspop_ours(pred_csv)
