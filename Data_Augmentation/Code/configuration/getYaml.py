import argparse
import pytorch_models.cifar as models
import yaml

def getYaml(path:str):
    with open(path) as f:
        cfg  = yaml.load(f, Loader=yaml.FullLoader)
    return cfg



