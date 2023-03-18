from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim, utils

from config import ModelConfig
from models.model_utils import VanilaCnnModel, fc_block
from typing import List

class VanillaFcnn(pl.LightningModule):
    def __init__(
        self, lr: float = 0.001, device: str = "cpu"
    ) -> None:
        """
        This model takes first correlated variable data, and given variable data, passes thru an MLP and 
        predicts the value of given variable for next timestep
        """
        super().__init__()
        self.save_hyperparameters()
        self._net = nn.Sequential(*fc_block(7,7),
                                  *fc_block(7,7),
                                  *fc_block(7,7),
                                  *fc_block(7,3),
                                  nn.Linear(3,1),)

    def to_batch(self, data: List[Tensor]) -> Tensor:
        """Convert a list of unequal rows of variables to a uniform batch, taking only the first variable"""
        return torch.stack(list(map(lambda x: x[0,None],data)))

    def training_step(self, batch, batch_idx) -> Tensor:
        x1, x2, y = batch
        y_hat = self._net(torch.hstack([x1[:,0,-3:], self.to_batch(x2)[:,0,-4:]]))
        loss = nn.functional.mse_loss(y_hat, y, reduction="mean")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x1, x2, y = batch
        y_hat = self._net(torch.hstack([x1[:,0,-3:], self.to_batch(x2)[:,0,-4:]]))
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)
    
    def test_step(self,  batch, batch_idx) -> None:
        x1, x2, y = batch
        y_hat = self._net(torch.hstack([x1[:,0,-3:], self.to_batch(x2)[:,0,-4:]]))

        y_up_down = torch.sign(y.squeeze())
        y_hat = torch.sign(y_hat)
        
        return (y_hat,y_up_down)
    
    def test_epoch_end(self, outputs) -> None:
        # Combine outputs from all batches
        y_hat = torch.stack(list(map(lambda x: x[0],outputs[:-1]))).flatten()
        y = torch.stack(list(map(lambda x: x[1],outputs[:-1]))).flatten()
            
        self.log("Accuracy",torch.sum(y_hat==y)/y.shape[0])
        return outputs
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
