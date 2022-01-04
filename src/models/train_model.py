# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from src.models.model import MyAwesomeModel
import tqdm
import matplotlib.pyplot as plt
import os
from src.data.make_dataset import load_data


@click.command()
@click.argument("lr", type=click.FLOAT)
@click.argument("epochs", type=click.INT)
def train(lr, epochs):
    print("Training day and night")

    model = MyAwesomeModel()
    train_set = load_data()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    for e in tqdm.tqdm(range(epochs), unit="epoch"):
        running_loss = 0
        for images, labels in trainloader:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss)

    num = len(os.listdir("src/models/trained_models"))
    torch.save(
        model.state_dict(), "src/models/trained_models/trained_model_{}.pth".format(num)
    )
    plt.plot(train_losses)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig("reports/figures/training_curve_{}.png".format(num))


def main():
    """Runs training loop"""
    logger = logging.getLogger(__name__)
    logger.info("train model")
    train()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
