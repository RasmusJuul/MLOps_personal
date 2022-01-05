# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.models.model import ResNet, CNN

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config.yaml")
def predict(cfg):
    logger.info((f"Configuration: \n {OmegaConf.to_yaml(cfg)}"))

    if cfg.training.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(cfg.predict.model_path)
    
    if checkpoint['model_type'].lower() == "cnn":
        model = CNN(features=checkpoint['features'],
                    height=28,
                    width=28,
                    droprate=cfg.model.droprate)
    elif checkpoint['model_type'].lower() == 'resnet':
        model = ResNet(features=checkpoint['features'],
                       height=28,
                       width=28,
                       droprate=checkpoint['droprate'],
                       num_blocks=checkpoint['num_blocks'])
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    try:
        images = np.load(cfg.predict.data_path)
    except OSError as err:
        logger.warning("OS error: {0}".format(err))
        return None

    images = torch.from_numpy(images).float()

    with torch.no_grad():
        model.eval()
        log_ps = model(images)
        ps = torch.exp(log_ps)
        _, top_class = ps.topk(1, dim=1)

    # fashion = {0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle Boot'}
    # predictions = [fashion[pred.item()] for pred in top_class]
    predictions = [pred.item() for pred in top_class]
    logger.info(predictions)


def main():
    """Creates prediction using pre-trained model"""
    logger = logging.getLogger(__name__)
    logger.info("use best pre-trained model to predict")
    predict()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
