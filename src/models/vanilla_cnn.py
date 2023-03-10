from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim, utils

from config import ModelConfig
from models.model_utils import VanilaCnnModel


class VanillaCnn(pl.LightningModule):
    def __init__(
        self, lr: float = 0.001, net: nn.Module = VanilaCnnModel(), device: str = "cpu"
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="net")
        self._net = net

    def training_step(self, batch, batch_idx) -> Tensor:
        x1, x2, y = batch
        y_hat = self._net(x1, x2)
        loss = nn.functional.mse_loss(y_hat, y, reduction="mean")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x1, x2, y = batch
        y_hat = self._net(x1, x2)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
