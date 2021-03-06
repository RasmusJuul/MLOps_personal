# -*- coding: utf-8 -*-
import glob
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Get file names
    train_files = glob.glob(input_filepath + "/train*.npz")
    test_files = glob.glob(input_filepath + "/test*.npz")

    # Extracts data from npz files
    images = []
    labels = []
    for file in train_files:
        data = np.load(file)
        images.append(data["images"])
        labels.append(data["labels"])
    train_images = np.concatenate((images), axis=0)
    train_labels = np.concatenate((labels), axis=0)

    images = []
    labels = []
    for file in test_files:
        data = np.load(file)
        images.append(data["images"])
        labels.append(data["labels"])
    test_images = np.concatenate((images), axis=0)
    test_labels = np.concatenate((labels), axis=0)

    # Cast to desired types
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).long()

    # Normalize
    train_images = torch.nn.functional.normalize(train_images)
    test_images = torch.nn.functional.normalize(test_images)

    # Add dimension
    train_images = torch.unsqueeze(train_images, dim=1)
    test_images = torch.unsqueeze(test_images, dim=1)

    # Save in processed folder
    torch.save(train_images, output_filepath + "/train_images.pt")
    torch.save(train_labels, output_filepath + "/train_labels.pt")
    torch.save(test_images, output_filepath + "/test_images.pt")
    torch.save(test_labels, output_filepath + "/test_labels.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
