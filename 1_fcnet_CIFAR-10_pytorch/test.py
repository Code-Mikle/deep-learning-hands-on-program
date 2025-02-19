"""
    测试代码
"""
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
import pandas as pd
from model import Classifier

conf_test = OmegaConf.load('config/config.yaml')
OmegaConf.to_yaml(conf_test, resolve=True)

def plot_loss_accuracy_from_csv():
    air_quality = pd.read_csv(conf_test.result_path + "loss_accuracy.csv", index_col=0)
    print(air_quality.head())
    air_quality['accuracy'].plot()
    plt.ylabel('Accuracy')
    plt.show()
    air_quality['loss'].plot()
    plt.ylabel('Loss')
    plt.show()


def load_parameters():
    _model = Classifier()
    _model.load_state_dict(torch.load(conf_test.result_path + 'parameter.pt', weights_only=True))


def load_model_parameters():
    _model = torch.load(conf_test.result_path + 'model_and_parameters.pt', weights_only=True, map_location=torch.device('cuda'))


if __name__ == '__main__':
    plot_loss_accuracy_from_csv()
