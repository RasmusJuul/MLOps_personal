# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

from src.models.model import MyAwesomeModel


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
def visualizations(model_path):
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path))

    images = torch.from_numpy(images).float()


def main():
    """Visualize"""
    logger = logging.getLogger(__name__)
    logger.info("save visualizations")
    visualizations()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
