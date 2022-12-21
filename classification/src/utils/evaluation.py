import os
from collections import OrderedDict
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.file_folder_ops import save_json
from utils.utils import get_fnames_labs_clf, get_cities, get_fnames_labs_citywise_clf
from utils.constants import config_path, img_rows, img_cols, all_patches_mixed_test_part1, all_patches_mixed_train_part1

from utils.dataset import PopulationDataset_Clf
from utils.metrics import save_clf_metrics, save_clf_metrics_citywise


def evaluate(model, model_name, exp_dir, osm_flag):
    '''
     Function to evaluate all test data
     ------------
     :Args:
         model: Pytorch Model Class to use for evaluation
         model_name: name of pytorch model to load parameters from
         exp_dir: where to save results
         osm_flag: if osm data should be used or not
     -----------
     :Return:
     - Function will create a evaluation of the the model on the test set. Results will be saved according to
     Function save_reg_metrics
     '''

    params = {'dim': (img_rows, img_cols), 'n_classes': 17}
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title = model_name
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name + '.pth'),
                            map_location=torch.device(map_location))
    model_scale = model_dict['hyperparams']['model_scale']
    num_class = model_dict['hyperparams']['num_classes']
    if osm_flag:
        model = model(input_channels=10, num_classes=num_class, scale_factor=model_scale)
    else:
        model = model(input_channels=10, num_classes=num_class)
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])
    ##  manually loading
    data_dir = all_patches_mixed_test_part1
    f_names_test, labels_test = get_fnames_labs_clf(data_dir)
    np.save(os.path.join(config_path, 'f_lists', 'f_names_test.npy'), f_names_test)
    np.save(os.path.join(config_path, 'f_lists', 'labels_test.npy'), labels_test)
    # to load the saved numpy array
    # f_names_test = np.load(os.path.join(config_path, 'f_lists', 'f_names_test.npy'))
    # labels_test = np.load(os.path.join(config_path, 'f_lists', 'labels_test.npy'))
    test_dataset = PopulationDataset_Clf(f_names_test, labels_test, test=True, **params)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False, num_workers=2)

    all_preds = torch.zeros((len(test_loader.dataset)))
    all_targets = torch.zeros((len(test_loader.dataset)))
    for i, data in enumerate(test_loader):  ## change to val or test
        model.eval()
        if osm_flag:
            inputs, targets, osm = data['input'].to(device), \
                                   data['label'].to(device), data['osm'].to(device)
            with torch.no_grad():
                preds = model(inputs, osm)
        else:
            inputs, targets = data['input'].to(device), data['label'].to(device)
            with torch.no_grad():
                preds = model(inputs)

        _, predicted = preds.max(1)
        _, labels = targets.max(1)
        all_preds[i * batch_size:i * batch_size + preds.shape[0]] = predicted
        all_targets[i * batch_size:i * batch_size + preds.shape[0]] = labels

    fol_path = os.path.join(exp_dir, 'log', title.replace('_model', ''))
    if not os.path.exists(fol_path):
        os.mkdir(fol_path)
    save_clf_metrics(all_targets, all_preds, fol_path, dataset='test')


def evaluate_citywise_id(model, model_name, exp_dir, osm_flag, data_dir):
    '''
    Function to evaluate data city wise and return IDs
    ------------
    :Args:
        model: Pytorch Model Class to use for evaluation
        model_name: name of pytorch model to load parameters from
        exp_dir: where to save results
        osm_flag: if osm data should be used or not
        data_dir: directory of data
    -----------
    :Return:
    - Function will create a evaluation of the the model on the test set. Results will be saved according to
    Function save_reg_metrics
    '''
    if data_dir == all_patches_mixed_test_part1:
        dataset_name = 'test'
    else:
        print('Train:', data_dir == all_patches_mixed_train_part1)
        dataset_name = 'train'

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
            model = model(input_channels=9, num_classes=num_class, scale_factor=model_scale)
        else:
            model = model(input_channels=10, num_classes=num_class, scale_factor=model_scale)
    else:
        model = model(input_channels=10, num_classes=1)
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])

    ## manually loading
    cities = get_cities(data_dir)

    x = []
    neg_macd5_greater = []
    neg_macd4 = []
    neg_macd3 = []
    neg_macd2 = []
    neg_macd1 = []
    macd0 = []
    macd1 = []
    macd2 = []
    macd3 = []
    macd4 = []
    macd5_greater = []
    macd6_greater = []

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
        f_names_test, labels_test = get_fnames_labs_citywise_clf(city)
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
        save_json(denormalized_dict, os.path.join(exp_dir, 'log', title, city_name, title + '_allpredictions_id.json'))
        save_clf_metrics_citywise(all_targets, all_preds, os.path.join(exp_dir, 'log', title, city_name), ID_list,
                                  dataset_name=dataset_name)

        total = np.array(all_targets).size
        cc = Counter(np.array(all_targets) - np.array(all_preds))
        # plots macd
        macd0_temp = []

        macd1_temp = []
        macd2_temp = []
        macd3_temp = []
        macd4_temp = []
        macd5_temp = []

        neg_macd1_temp = []
        neg_macd2_temp = []
        neg_macd3_temp = []
        neg_macd4_temp = []
        neg_macd5_temp = []

        for key, value in enumerate(cc.items()):
            if value[0] == 0.0:
                macd0_temp.append(value[1])
            elif value[0] == 1.0:
                macd1_temp.append(value[1])
            elif value[0] == 2.0:
                macd2_temp.append(value[1])
                '''
                elif value[0] == 3.0:
                    macd3_temp.append(value[1])
                elif value[0] == 4.0:
                    macd4_temp.append(value[1])
                '''

            elif value[0] == -1.0:
                neg_macd1_temp.append(value[1])
            elif value[0] == -2.0:
                neg_macd2_temp.append(value[1])
                '''
                elif value[0] == -3.0:
                    neg_macd3_temp.append(value[1])
                elif value[0] == -4.0:
                    neg_macd4_temp.append(value[1])
                '''
            else:
                if value[0] > 0:
                    macd5_temp.append(value[1])

                if value[0] < 0:
                    neg_macd5_temp.append(value[1])

        if len(macd0_temp) > 0:
            macd0.append((macd0_temp[0]/total)*100)
        else:
            macd0.append(0 / total)

        if len(macd1_temp) > 0:
            macd1.append((macd1_temp[0] / total)*100)
        else:
            macd1.append(0 / total)

        if len(macd2_temp) > 0:
            macd2.append((macd2_temp[0] / total)*100)
        else:
            macd2.append(0 / total)
            '''
            if len(macd3_temp) > 0:
                macd3.append(macd3_temp[0] / total)
            else:
                macd3.append(0 / total)
    
            if len(macd4_temp) > 0:
                macd4.append(macd4_temp[0] / total)
            else:
                macd4.append(0 / total)
            '''
        if len(neg_macd1_temp) > 0:
            neg_macd1.append((neg_macd1_temp[0] / total)*100)
        else:
            neg_macd1.append(0 / total)

        if len(neg_macd2_temp) > 0:
            neg_macd2.append((neg_macd2_temp[0] / total)*100)
        else:
            neg_macd2.append(0 / total)
            '''
            if len(neg_macd3_temp) > 0:
                neg_macd3.append(neg_macd3_temp[0] / total)
            else:
                neg_macd3.append(0 / total)
    
            if len(neg_macd4_temp) > 0:
                neg_macd4.append(neg_macd4_temp[0] / total)
            else:
                neg_macd4.append(0 / total)
            '''
        if len(macd5_temp) > 0:
            macd5_greater.append(((np.sum(macd5_temp)) / total)*100)
        else:
            macd5.append(0 / total)

        if len(neg_macd5_temp) > 0:
            neg_macd5_greater.append(((np.sum(neg_macd5_temp)) / total)*100)
        else:
            neg_macd5_greater.append(0 / total)

        x.append(city_name.rsplit('_')[-1])

    y0 = np.array(neg_macd1)
    y1 = np.array(neg_macd2)
    #y2 = np.array(neg_macd3)
    #y3 = np.array(neg_macd4)
    y4 = np.array(neg_macd5_greater)

    y5 = np.array(macd0)
    y6 = np.array(macd1)
    y7 = np.array(macd2)
    #y8 = np.array(macd3)
    #y9 = np.array(macd4)
    y10 = np.array(macd5_greater)

    # plot bars in stack manner
    width = 0.25
    fig, ax = plt.subplots()

    ax.bar(x, y4, color='y', width=width)
    ax.bar(x, y1, bottom=y4, color='g', width=width)
    ax.bar(x, y0, bottom=y4 + y1, color='b', width=width)
    ax.bar(x, y5, bottom=y4 + y1 + y0, color='r', width=width)
    ax.bar(x, y6, bottom=y4 + y1 + y0 + y5, color='b', width=width)
    ax.bar(x, y7, bottom=y4 + y1 + y0 + y5 + y6, color='g', width=width)
    ax.bar(x, y10, bottom=y4 + y1 + y0 + + y5 + y6 + y7, color='y', width=width)

    ax.set_xlabel("City")
    ax.set_ylabel("Percentage of test patches")

    ax.legend(["MACD >= -3", "MACD = -2", "MACD = -1", "MACD = 0", "MACD = 1", "MACD = 2", "MACD >= 3"], ncol=2, loc=[1,0])
    #ax.set_title("MACD distribution")
    plt.show()
    path = os.path.join(exp_dir, 'log', title, city_name, title + '_macd.png')
    plt.savefig(path, bbox_inches="tight", dpi=600)
