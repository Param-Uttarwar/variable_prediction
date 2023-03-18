import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

from config import Config
from src.data_ops.variable_dataset import VariableDataset
from src.data_ops.data_utils import collate
from src.models.vanilla_cnn import VanillaCnn
from src.models.vanilla_fcnn import VanillaFcnn
from src.models.baseline_model import BaselineModel
from src.models.pool_fcnn import PoolFcnn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CONFIG_PATH = Path("var/eval_conf.yaml")
torch.manual_seed(0)

def main() -> None:

    # Get Config
    config = Config.from_yaml(CONFIG_PATH)
    if config.model_filepath is None:
        raise ValueError('Model checkpoint path is not set')

    # Get dataset/dataloaders
    dataset = VariableDataset(device=config.device)
    dataset.load_from_csv(Path(config.data_filepath))
    train_data, val_data = random_split(dataset, [0.8, 0.2])
    train_dl = DataLoader(train_data, batch_size=config.batch_size, collate_fn=collate)
    val_dl = DataLoader(val_data, batch_size=config.batch_size, collate_fn=collate)

    # Test Base model
    print("Baseline accuracy")
    model = BaselineModel()
    trainer = pl.Trainer(logger=False)
    loss = trainer.test(model = model,
                 dataloaders=val_dl)
    

    # Test Pool FCNN model
    print("Pool FCNN Model accuracy")
    model = PoolFcnn(lr=config.model_config.lr, device=config.device)
    trainer = pl.Trainer(logger=False)
    loss = trainer.test(model = model,
                 dataloaders=val_dl,
                 ckpt_path='lightning_logs/version_0/checkpoints/epoch=13-step=6468.ckpt')
    



if __name__ == "__main__":
    main()