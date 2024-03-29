import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
import logging

import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

from config import Config
from src.data_ops.variable_dataset import VariableDataset
from src.data_ops.data_utils import collate
from src.models.vanilla_cnn import VanillaCnn
from src.models.vanilla_fcnn import VanillaFcnn
from src.models.pool_fcnn import PoolFcnn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CONFIG_PATH = Path("var/conf.yaml")


def main() -> None:
    # Get Config
    config = Config.from_yaml(CONFIG_PATH)

    # Get dataset/dataloaders
    dataset = VariableDataset(device=config.device)
    dataset.load_from_csv(Path(config.data_filepath))
    train_data, val_data = random_split(dataset, [0.8, 0.2])
    train_dl = DataLoader(train_data, batch_size=config.batch_size, collate_fn=collate)
    val_dl = DataLoader(val_data, batch_size=config.batch_size, collate_fn=collate)

    # Get model
    # model = VanillaCnn(lr=config.model_config.lr, device=config.device)
    # model = VanillaFcnn(lr=config.model_config.lr, device=config.device)
    model = PoolFcnn(lr=config.model_config.lr, device=config.device)

    # Load saved model
    if config.model_filepath is not None:
        model = model.load_from_checkpoint(Path(config.model_filepath))
        logger.info(f"Model loaded from {config.model_filepath}")

    # Run Trainer
    trainer = pl.Trainer(max_epochs=20, logger=True)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
