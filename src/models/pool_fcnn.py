from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim, utils

from config import ModelConfig
from models.model_utils import VanilaCnnModel, fc_block
from typing import List

class PoolFcnn(pl.LightningModule):
    def __init__(
        self, lr: float = 0.001, device: str = "cpu"
    ) -> None:
        """
        This model has a common MLP which processing correlated variable individually. Intermediate features
        are max pooled so its permutation invariant, Final MLP takes in these features and outputs single 
        value of given variable for next timestep
        """

        super().__init__()
        self.save_hyperparameters()
        self._pool_fc = nn.Sequential(*fc_block(7,7),
                                  nn.Linear(7,7))
        self._final_fc = nn.Sequential(*fc_block(7,7),
                                       nn.Linear(7,3),
                                       nn.Linear(3,1))

    def batch_process(self, x1:Tensor, x2: List[Tensor]) -> Tensor:
        """
        Returns the output prediction of all datapoints in a batch. Needs to process indiviually since x2 is
        a list of different number of correlated variables which needs to be max pooled differently for each sample
        """

        if x1.shape[0]!=len(x2):
            raise ValueError('Shape mismatch')
        batch_size = x1.shape[0]
        y_hat = []

        for idx in range(batch_size):
            x1x2 = [torch.hstack([x1[idx,0,-3:], x2[idx][i,-4:]]) for i in range(x2[idx].shape[0])]
            x2_fc = list(map(self._pool_fc,x1x2))
            x2_fc = torch.max(torch.stack(x2_fc),dim=0).values
            y_hat.append(self._final_fc(x2_fc)) 
        
        return torch.stack(y_hat).squeeze()

    def training_step(self, batch, batch_idx) -> Tensor:
        x1, x2, y = batch
        y_hat = self.batch_process(x1,x2)
        loss = nn.functional.mse_loss(y_hat, y.squeeze())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x1, x2, y = batch
        y_hat = self.batch_process(x1,x2)
        loss = nn.functional.mse_loss(y_hat, y.squeeze())
        self.log("val_loss", loss)
        return loss
    
    def test_step(self,  batch, batch_idx) -> None:
        x1, x2, y = batch
        y_hat = self.batch_process(x1,x2)
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
