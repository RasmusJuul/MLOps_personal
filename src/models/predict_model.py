# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from src.models.model import MyAwesomeModel
import numpy as np


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True)) #Should be a numpy file with an array of images with shape [num,height,width]
def predict(model_path, data_path):
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path))
    try:
        images = np.load(data_path)
    except OSError as err:
        print("OS error: {0}".format(err))
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
    print(predictions)




def main():
    """ Creates prediction using pre-trained model
    """
    logger = logging.getLogger(__name__)
    logger.info('use pre-trained model to predict')
    predict()





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()