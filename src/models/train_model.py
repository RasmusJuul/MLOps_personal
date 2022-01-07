# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import tqdm
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from src.data.load_data import load_data
from src.models.model import ResNet, CNN

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config.yaml")
def train(cfg: DictConfig):
    logger.info((f"Configuration: \n {OmegaConf.to_yaml(cfg)}"))
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
    logger.info(f"Running on {device}")

    if cfg.model.model_type.lower() == "cnn":
        model = CNN(features=cfg.model.features,
                    height=28,
                    width=28,
                    droprate=cfg.model.droprate)
    elif cfg.model.model_type.lower() == 'resnet':
        model = ResNet(features=cfg.model.features,
                       height=28,
                       width=28,
                       droprate=cfg.model.droprate,
                       num_blocks=cfg.model.num_blocks)
    else:
        logger.error("Invalid model chosen")
        return None
    model.to(device)
    model.train()

    train_set,_ = load_data()
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size, shuffle=True
    )
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    train_losses = []
    train_acc = []
    for e in tqdm.tqdm(range(cfg.training.epochs), unit="epoch"):
        running_loss = 0
        train_correct = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            log_ps = model(images)
            # Compute the loss
            loss = criterion(log_ps, labels)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            running_loss += loss.item()

            # Compute how many were correctly classified
            predicted = torch.exp(log_ps).argmax(1)
            train_correct += (labels == predicted).sum().cpu().item()

            # Remove mini-batch from memory
            del images, labels
        train_losses.append(running_loss)
        train_acc.append(train_correct / len(trainloader.dataset))

    os.makedirs("../../../models/", exist_ok=True)
    date = datetime.today().strftime('%d-%m-%y:%H%M')
    checkpoint = {'model_type':cfg.model.model_type,
                  'features':cfg.model.features,
                  'droprate':cfg.model.droprate,
                  'num_blocks':cfg.model.num_blocks,
                  'state_dict':model.state_dict()}
    torch.save(
        checkpoint, "../../../models/trained_model_{}.pth".format(date)
    )
    logger.info("Model saved")
    os.makedirs("../../../reports/figures/", exist_ok=True)
    plt.plot(train_losses)
    plt.title("Training loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig("../../../reports/figures/training_curve_{}.png".format(date))
    plt.figure()
    plt.plot(train_acc)
    plt.title("Training accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.savefig("../../../reports/figures/training_acc_{}.png".format(date))
    logger.info("Graphs saved")


def main():
    """Runs training loop"""
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
