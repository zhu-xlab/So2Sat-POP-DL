import os
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
import seaborn as sns
from utils.file_folder_ops import save_json
from utils.constants import img_rows, img_cols,all_patches_mixed_test_part1, all_patches_mixed_train_part1, \
    osm_prominent_feature_names, fig_size, fig_size_heatmap, osm_feature_names
from utils.utils import get_fnames_labs_reg, get_cities, \
    get_fnames_labs_citywise_reg
from utils.dataset import PopulationDataset_Reg, denormalize_reg_labels
from utils.metrics import save_reg_metrics
from utils.utils_explainability import convert_to_viz_format, normalize_image, plot_palette
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt


def evaluate(model, model_name, exp_dir, osm_flag):
    """
    Function to evaluate all cities at once
    ------------
    :Args:
        model: Pytorch Model Class to use for evaluation
        model_name: name of pytorch model to load parameters from
        exp_dir: where to save results
        osm_flag: if osm data should be used or not
        data_dir: directory of data
    -----------
    :Return:
    - Function will create a evaluation of the the model on the test set.
    Results will be saved according to Function save_reg_metrics
    """
    id_list = []
    params = {'dim': (img_rows, img_cols)}
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    title = model_name
    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name + '.pth'), map_location=torch.device(map_location))
    model_scale = model_dict['hyperparams']['model_scale']
    if osm_flag is True:
        model = model(input_channels=10, num_classes=1, scale_factor=model_scale)
    else:
        model = model(input_channels=10, num_classes=model_scale)
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])
    ## manually loading
    data_dir = all_patches_mixed_test_part1
    f_names_test, labels_test = get_fnames_labs_reg(data_dir)
    #f_names_test = np.load(os.path.join(config_path, 'f_lists', 'f_names_test.npy'))
    #labels_test = np.load(os.path.join(config_path, 'f_lists', 'labels_test.npy'))
    test_dataset = PopulationDataset_Reg(f_names_test, labels_test, test=True, **params)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)
    all_preds = torch.zeros((len(test_dataset)))
    all_targets = torch.zeros((len(test_dataset)))
    assert len(all_preds) == len(f_names_test)
    for i, data in enumerate(test_loader): ## change to val or test
        model.eval()
        if osm_flag is True:
            inputs, targets, osm, ID = data['input'].to(device), \
                data['label'].to(device), data['osm'].to(device), data['identifier'][0]
            with torch.no_grad():
                preds = model(inputs, osm)
        else:
            inputs, targets, ID = data['input'].to(device), data['label'].to(device), data['identifier'][0]
            with torch.no_grad():
                preds = model(inputs)
        preds = preds.view(-1)
        all_preds[i*batch_size:i*batch_size+preds.shape[0]] = preds
        all_targets[i*batch_size:i*batch_size+preds.shape[0]] = targets
        id = ID.rsplit('/')[-1]
        id_list.append(id)
        if not os.path.exists(os.path.join(exp_dir, 'log', title.replace('_model', ''))):
            os.mkdir(os.path.join(exp_dir, 'log', title.replace('_model', '')))

    save_reg_metrics(all_targets, all_preds, id_list, os.path.join(exp_dir, 'log', title.replace('_model', '')), dataset_name='test')
    print('Logs saved at:', os.path.join(exp_dir, 'log', title.replace('_model', '')))


def evaluate_citywise_id(model, model_name, exp_dir, osm_flag, data_dir):
    """
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
    - Function will create a evaluation of the the model on the test set.
    Results will be saved according to Function save_reg_metrics
    """
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
    model_dict = torch.load(os.path.join(exp_dir, 'models', model_name+'.pth'), map_location=torch.device(map_location))
    model_scale = model_dict['hyperparams']['model_scale']
    if osm_flag is True:
        if 'viirs' in model_name:
            print('Evaluation without VIIRS')
            model = model(input_channels=9, num_classes=1, scale_factor=model_scale)
        else:
            model = model(input_channels=10, num_classes=1, scale_factor=model_scale)
    else:
        model = model(input_channels=10, num_classes=1)
    model.to(device)
    model.load_state_dict(model_dict['model_state_dict'])
    ## manually loading
    cities = get_cities(data_dir)
    explainability_results_root_dir = os.path.join(exp_dir, "explainability_results")
    if not os.path.exists(explainability_results_root_dir):
        os.makedirs(explainability_results_root_dir)

    osm_color_palette = sns.color_palette("Paired", len(osm_feature_names))
    osm_color_palette_dict = dict(zip(osm_feature_names, osm_color_palette))

    fig_pal, ax_pal = plot_palette(osm_color_palette)
    fig_pal.savefig(os.path.join(explainability_results_root_dir, "color_palette.png"))

    integrated_gradients = IntegratedGradients(model)
    for city in cities:
        all_targets = torch.empty(0)
        all_preds = torch.empty(0)
        denormalized_dict = OrderedDict()
        denormalized_dict['preds'] = {}
        denormalized_dict['targets'] = {}
        id_list = []
        print(city)
        city_name = os.path.basename(city)

        explainability_results_dir = os.path.join(explainability_results_root_dir, city_name)
        if not os.path.exists(explainability_results_dir):
            os.makedirs(explainability_results_dir)

        denormalized_dict['preds'][city_name] = {}
        denormalized_dict['targets'][city_name] = {}
        f_names_test, labels_test = get_fnames_labs_citywise_reg(city)
        test_dataset = PopulationDataset_Reg(f_names_test, labels_test, mode=model_name, test=True, **params)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        city_preds = torch.zeros((len(test_dataset)))
        city_targets = torch.zeros((len(test_dataset)))
        for i, data in enumerate(test_loader):
            model.eval()
            if osm_flag is True:
                inputs, targets, osm, ID = data['input'].to(device), \
                    data['label'].to(device), data['osm'].to(device), data['identifier'][0]
                with torch.no_grad():
                    preds = model(inputs, osm)
            else:
                inputs, targets, ID = data['input'].to(device), data['label'].to(device), data['identifier'][0]
                with torch.no_grad():
                    preds = model(inputs)
            id_list.append(ID.rsplit('/')[-1])
            preds = preds.view(-1)

            # create the explanation dir
            explanation_result_path = os.path.join(explainability_results_dir, ID)
            if not os.path.exists(explanation_result_path):
                os.makedirs(explanation_result_path)
            print('Explainability logs saved at', explanation_result_path)

            # generate explanations
            attributions_ig = integrated_gradients.attribute((inputs, osm), n_steps=50)
            attributions_osm = attributions_ig[1]

            attributions_osm = np.abs(torch.squeeze(attributions_osm).cpu().detach().numpy())
            if np.any(attributions_osm):
                attributions_osm = (attributions_osm - attributions_osm.min()) / (
                            attributions_osm.max() - attributions_osm.min())

                idx_sorted = np.flip(np.argsort(attributions_osm)[-5:])
                top_osm_features = attributions_osm[idx_sorted]

                top_feature_names = np.array(osm_feature_names)[idx_sorted]
                print(top_feature_names)

                fig_osm, ax_osm = plt.subplots(figsize=fig_size, nrows=1, ncols=1)
                sns.barplot(x=top_feature_names, y=top_osm_features, ax=ax_osm, hue=top_feature_names,
                            palette=osm_color_palette_dict)
                ax_osm.tick_params(axis="both", labelsize=12)

                ax_osm.spines['right'].set_visible(False)
                ax_osm.spines['top'].set_visible(False)

                ax_osm.axes.get_xaxis().set_visible(False)
                ax_osm.get_legend().remove()
                fig_osm.savefig(os.path.join(explanation_result_path, "osm_features.png"), bbox_inches='tight')

            attributions_image = convert_to_viz_format(attributions_ig[0])
            image_in_viz_format = convert_to_viz_format(inputs)
            sen2_image = image_in_viz_format[:, :, 0:3]

            fig_combined_viz, axs_combined_viz = viz.visualize_image_attr(attributions_image, None, method="heat_map",
                                                                          show_colorbar=False, use_pyplot=False,
                                                                          fig_size=fig_size_heatmap)

            fig_sen_2, axs_sen_2 = viz.visualize_image_attr(None, normalize_image(sen2_image), method="original_image",
                                                            use_pyplot=False, fig_size=fig_size)

            fig_combined_viz.savefig(os.path.join(explanation_result_path, "heatmap_all_channels_combined.png"))
            fig_sen_2.savefig(os.path.join(explanation_result_path, "sen2_image.png"))

            plt.close("all")

            city_preds[i*batch_size:i*batch_size+preds.shape[0]] = preds
            city_targets[i*batch_size:i*batch_size+preds.shape[0]] = targets

            denormalized_dict['preds'][city_name][ID] = denormalize_reg_labels(preds.item())
            denormalized_dict['targets'][city_name][ID] = targets.item()

        all_targets = torch.cat([all_targets, city_targets])
        all_preds = torch.cat([all_preds, city_preds])

        if not os.path.exists(os.path.join(exp_dir, 'log', title)):
            os.makedirs(os.path.join(exp_dir, 'log', title))
        if not os.path.exists(os.path.join(exp_dir, 'log', title, city_name)):
            os.mkdir(os.path.join(exp_dir, 'log', title, city_name))

        save_json(denormalized_dict, os.path.join(exp_dir, 'log', title, city_name, title + '_allpredictions_id.json'))
        save_reg_metrics(all_targets, all_preds, id_list, os.path.join(exp_dir, 'log', title, city_name),
                         dataset_name=dataset_name)
