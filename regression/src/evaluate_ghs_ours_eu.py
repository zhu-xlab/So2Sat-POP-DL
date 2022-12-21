# compares our predictions with ghs-pop

import glob
import os

import fiona
import torch
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import DataLoader
from utils.utils import get_cities
from shapely import wkt
from utils.constants import img_rows, img_cols, current_dir_path, all_patches_mixed_test_part1, exp_path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from collections import OrderedDict
from utils.metrics import plot_scatter
from utils.dataset import PopulationDataset_Reg, denormalize_reg_labels
from utils.file_folder_ops import save_json
from models.regression import EO2ResNet_OSM


def plot_by_pop(pop_shp_path, plot_path, csv_path, title, column_name, gt_max):
    """
      plots the population count using shape file of the city
    """
    pop_shp = fiona.open(pop_shp_path)
    crs = pop_shp.crs
    df = pd.read_csv(csv_path)
    df['geometry'] = df['geometry'].apply(wkt.loads)
    # df = df[df.PR_POP != 0] # omitting zero population count
    gdf = gpd.GeoDataFrame(df, crs=crs)
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(1, 1)
    ax.set(title=title)
    gdf.plot(column=column_name, legend=True, figsize=(10, 10), cmap='jet', vmin=0, vmax=gt_max)
    plt.axis('off')
    plt.savefig(plot_path)


def get_fnames_labs_eu(path):
    """
    :param path: path to patch folder (sen2spring)
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
        labs = [pop[index]]
        labs_all = np.append(labs_all, labs, axis=0)

    return f_names_all, labs_all


def predict_eu_cities(model, model_name, exp_dir, osm_flag, data_dir):
    """
    Predicts on the EU cities
    Args:
        model= trained model to be used
        model_name = name of the trained model
        exp_dir = path to result directory
        osm_flag = flag to use OSM data or not
        data_dir = path to the data directory
    Returns the path to prediction csv
    """
    dataset_name = 'eu'
    params = {'dim': (img_rows, img_cols)}
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title = model_name
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name + '.pth'),
                            map_location=torch.device(map_location))
    model_scale = model_dict['hyperparams']['model_scale']
    num_class = 1
    if osm_flag:
        if 'viirs' in model_name:
            print('Evaluation without VIIRS')
            model = model(input_channels=10, num_classes=num_class, scale_factor=model_scale)
        else:
            model = model(input_channels=10, num_classes=num_class, scale_factor=model_scale)
    else:
        model = model(input_channels=10, num_classes=num_class)
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
        f_names_test, labels_test = get_fnames_labs_eu(city)
        test_dataset = PopulationDataset_Reg(f_names_test, labels_test, test=True,**params)
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
            preds = preds.view(-1)

            city_preds[i * batch_size:i * batch_size + preds.shape[0]] = denormalize_reg_labels(preds.item())
            city_targets[i * batch_size:i * batch_size + preds.shape[0]] = targets
            ID_list.append(ID)
            denormalized_dict['preds'][city_name][ID] = denormalize_reg_labels(preds.item())
            denormalized_dict['targets'][city_name][ID] = targets.item()  # .cpu().numpy()#.detach().numpy()
        all_targets = torch.cat([all_targets, city_targets])
        all_preds = torch.cat([all_preds, city_preds])
        if not os.path.exists(os.path.join(exp_dir, 'log', title)):
            os.makedirs(os.path.join(exp_dir, 'log', title))
        if not os.path.exists(os.path.join(exp_dir, 'log', title, city_name)):
            os.mkdir(os.path.join(exp_dir, 'log', title, city_name))
        save_json(denormalized_dict,
                  os.path.join(exp_dir, 'log', title, city_name, title + '_allpredictions_id_final_reg_eu.json'))
        df = pd.DataFrame()
        df['GRD_ID'] = ID_list
        df['PR_POP'] = all_preds.tolist()
        df['GT_POP'] = all_targets.tolist()
        pred_csv = os.path.join(exp_dir, 'log', title, city_name, title + '_evaluation_final_eu.csv')
        df.to_csv(pred_csv, index=False)
        return pred_csv


def evaluate_ghspop_ours(pred_csv_path):
    """
    evaluates the meterics, compares with GHS-POP and
    Args:
        path to our predictions csv file
    """
    city_rasters = os.path.join(current_dir_path, 'city_rasters/ghs_vs_ours_eu/')
    all_cities = glob.glob(os.path.join(city_rasters, '*'))
    for city_path in all_cities:
        current_city_rasters = os.path.join(city_rasters, city_path.rsplit('\\')[-1:][0])
        eu_raster_path = glob.glob(os.path.join(current_city_rasters, '*eu_count.tif'))[0]
        ghs_raster_path = glob.glob(os.path.join(current_city_rasters, '*ghs_pop_count.tif'))[0]
        log_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_reg_log.txt')

        # load & plot Ground Truth
        with rasterio.open(os.path.join(eu_raster_path), 'r') as ds:
            image = ds.read()
        gt_max = image.max()

        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(10, 10))
        image = np.ma.array(image, mask=image < 0)
        m = plt.imshow(image.transpose(1, 2, 0), cmap='jet', vmin=0, vmax=gt_max)
        plt.colorbar(m, shrink=1.0)
        plt.axis('off')
        fig.savefig(os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_reg_gt.png'), bbox_inches='tight')
        plt.show()

        with rasterio.open(ghs_raster_path, 'r') as ds:
            ghs_image = ds.read()

        ## load & plot GHS POP
        ghs_image = np.ma.array(ghs_image, mask=image < 0)
        ghs_arr = np.where(ghs_image.compressed() < 0, 0, ghs_image.compressed())

        fig = plt.figure(figsize=(10, 10))
        ghs_image = np.ma.array(ghs_image, mask=image < 0)
        m = plt.imshow(ghs_image.transpose(1, 2, 0), cmap='jet', vmin=0, vmax=image.max())  # , cmap='pink')
        plt.colorbar(m, shrink=1.0)
        plt.axis('off')
        fig.savefig(os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_reg_ghs.png'), bbox_inches='tight')
        plt.show()

        ## Get Metrics
        RMSE = np.sqrt(mean_squared_error(image.compressed(), ghs_arr))
        MAE = mean_absolute_error(image.compressed(), ghs_arr)
        R2 = r2_score(image.compressed(), ghs_arr)

        with open(log_path, 'a') as f:
            f.writelines(
                "\n\n GHS POP results: \n — rmse: {} \n  — mae: {} \n — r2: {} \n".format(RMSE, MAE, R2))

        pop_shp_path = glob.glob(os.path.join(current_city_rasters, '*nonzero_geometries.shp'))[0]
        df_pop = gpd.read_file(pop_shp_path)
        df = pd.read_csv(pred_csv_path)

        id_list = list(df['GRD_ID'])
        ids = [x.rsplit('/')[-1:][0].rsplit('_')[0] for x in id_list]
        df['GRD_ID'] = ids
        gt_pop_list = df['GT_POP']
        pr_pop_list = df['PR_POP']
        for i in range(0, len(pr_pop_list)):
            if pr_pop_list[i] < 0:
                pr_pop_list[i] = 0

        df_sorted = pd.DataFrame()
        for i in range(0, len(df_pop.GRD_ID)):
            id = df_pop.GRD_ID[i]
            geometry = df_pop.geometry[i]
            row = df[df.GRD_ID == id]
            row['geometry'] = geometry
            df_sorted = pd.concat([df_sorted, row], ignore_index=True)

        ## Get Metrics
        RMSE = np.sqrt(mean_squared_error(df_sorted['GT_POP'].to_list(), df_sorted['PR_POP'].to_list()))
        MAE = mean_absolute_error(df_sorted['GT_POP'].to_list(), df_sorted['PR_POP'].to_list())
        R2 = r2_score(df_sorted['GT_POP'].to_list(), df_sorted['PR_POP'].to_list())

        with open(log_path, 'a') as f:
            f.writelines(
                "\n\n Our results: \n — rmse: {} \n  — mae: {} \n — r2: {} \n".format(RMSE, MAE, R2))

        csv_local_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_reg_evaluation_local.csv')
        df_sorted.to_csv(csv_local_path, index=False)

        pr_plot_path = os.path.join(city_path, city_path.split(os.sep)[-1:][0] + '_reg_pr_plot.png')
        plot_by_pop(pop_shp_path, pr_plot_path, csv_local_path, title='Predicted population', column_name='PR_POP',
                    gt_max=gt_max)

        plot_scatter(df_sorted['GT_POP'], df_sorted['PR_POP'], log_path.replace('.txt', '_scatter.png'))
        print('Logs saved at: ', log_path)


if __name__ == '__main__':
    model_name = 'reg_best_model'
    pred_csv = predict_eu_cities(model=EO2ResNet_OSM, model_name=model_name, exp_dir=exp_path, osm_flag=True,
                      data_dir=all_patches_mixed_test_part1)
    evaluate_ghspop_ours(pred_csv)
