#  Evaluation Skript for Test Dataset So2Sat POP

import argparse

from utils.evaluation import evaluate, evaluate_citywise_id
from utils.constants import exp_path, all_patches_mixed_test_part1
from models.classification import EO2ResNet_OSM, EOResNet


def parse():
    parser = argparse.ArgumentParser(description='Evaluation Skript Population Estimation')
    parser.add_argument('-m', '--model-name', type=str, help='name of model to evaluate', required=True)
    parser.add_argument('-c', '--city-wise', help='If set, evaluation will be citywise', default=True,
                        action='store_true')
    parser.add_argument('-o', '--no-osm', help='If set, Evaluation Skript wont use OSM Data', default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    model_name = args.model_name

    if args.no_osm:
        #  without OSM branch
        osm = False
        model = EOResNet
    else:
        # with OSM branch
        osm = True
        model = EO2ResNet_OSM
    if args.city_wise:
        # evaluates the model citywise
        evaluate_citywise_id(model=model, model_name=model_name, exp_dir=exp_path, osm_flag=osm,
                             data_dir=all_patches_mixed_test_part1)
    else:
        # evalautes the model on all the cities together
        evaluate(model=model, model_name=model_name, exp_dir=exp_path, osm_flag=osm)
