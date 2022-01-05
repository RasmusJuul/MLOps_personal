# -*- coding: utf-8 -*-
import logging
import glob
from pathlib import Path
from shutil import copy2

import hydra
import torch
import json
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.data.load_data import load_data
from src.models.model import ResNet, CNN

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config.yaml")
def test(cfg: DictConfig):
    logger.info((f"Configuration: \n {OmegaConf.to_yaml(cfg)}"))

    # Create dictionary with model filename as the key with the item being the full path
    models_dict = {file[16:-4]:file for file in glob.glob('../../../models/*.pth')}
    # Loads the json file acc which has the test accuracy for all models
    with open('../../../models/acc.json', 'r') as fp:
        acc = json.load(fp)

    # Set seed for everything
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if cfg.training.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in models_dict.keys():
        if model_name not in acc.keys():
            logger.info(f'Testing: {model_name}')
            # Load model checkpoint
            checkpoint = torch.load(models_dict[model_name])
            # Initialize model
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
            model.eval()
            # Load test dataset
            _,test_set = load_data()
            testloader = torch.utils.data.DataLoader(
                test_set, batch_size=cfg.training.batch_size, shuffle=True
            )

            # Run test
            res = []
            with torch.no_grad():
                for images, labels in testloader:
                    log_ps = model(images)
                    ps = torch.exp(log_ps)
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    res.append(equals)

                equals = torch.cat(res)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                logger.info(f'Accuracy: {accuracy.item()*100}%')
            acc[model_name] = accuracy.item()*100

    with open('../../../models/acc.json', 'w') as fp:
        json.dump(acc, fp)
    
    copy2(models_dict[max(acc,key=acc.get)],'../../../models/best/trained_model_best.pth')



def main():
    """Runs training loop"""
    logger.info("test model and compare to current best model")
    test()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()



