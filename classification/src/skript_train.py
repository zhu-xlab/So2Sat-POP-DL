# starts the training for the classification

import os
import argparse

from  utils.solver import Classification_Solver as Solver
from  utils.constants import config_path, exp_path
from models.classification import EO2ResNet_OSM, EOResNet
from utils.loss import FocalLoss
from utils.file_folder_ops import load_json


def parse():
    parser = argparse.ArgumentParser(description='Training Skript Population Estimation')
    parser.add_argument('-j', '--name-json', type=str, help='name of json_file containing the hyperparameters',
                        default="standard")
    parser.add_argument('-o', '--no-osm', help='If set, Training Skript wont use OSM Data', default=False,
                        action='store_true')
    parser.add_argument('-b', '--bal-acc', help='If set, Training use balanced accuracy', default=True,
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    input_name = args.name_json
    if args.no_osm:
        osm = False
    else:
        osm = True
    if args.bal_acc:
        print('Using balanced accuracy')

    #  directory for json contating the hyperparameters
    hparams_dict = load_json(os.path.join(config_path, 'hparams_dir', input_name + '.json'))

    if osm is True:
        print('Use of OSM Data')
        model = EO2ResNet_OSM
    else:
        print('Not using OSM Data')
        model = EOResNet

    loss_fct = FocalLoss

    solver = Solver(hparams_dict, exp_dir=exp_path, model=model, osm=osm, loss_fct=loss_fct, bal_acc=args.bal_acc)
    solver.train()
