# Training Skript

import os
import argparse

from torch import nn
from utils.solver import Regression_Solver as Solver
from utils.constants import config_path, exp_path
from models.regression import EO2ResNet_OSM, EOResNet
from utils.file_folder_ops import load_json

def parse():
    parser = argparse.ArgumentParser(description='Training Skript Population Estimation')
    parser.add_argument('-j', '--name-json', type=str, help='name of json_file containing the hyperparameters',
                        default="standard")
    parser.add_argument('-o', '--no-osm', help='If set, Training Skript wont use OSM Data', default=False,
                        action='store_true')
    parser.add_argument('-m', '--model-name', type=str, help='if argument is given, skript will continue training '
                                                             'given model\
        ; argument should be name of the model to be trained')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    input_name = args.name_json
    if args.no_osm:
        osm = False
    else:
        osm = True
    if args.model_name:
        model_name = args.model_name
        continue_train = True
    else:
        model_name = None
        continue_train = False

    #  directory for json contating the hyperparameters, change accordingly
    hparams_dict = load_json(os.path.join(config_path, 'hparams_dir', input_name + '.json'))
    hparams_dict['title'] = input_name
    if osm is True:
        print('Use of OSM Data')
        model = EO2ResNet_OSM
    else:
        print('Not using OSM Data')
        model = EOResNet
    loss_fct = nn.MSELoss
    solver = Solver(hparams_dict, 
                    exp_dir=exp_path, 
                    model=model, 
                    osm=osm, 
                    loss_fct=loss_fct, 
                    continue_train=continue_train, 
                    model_name=model_name)
    solver.train()
