from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim, utils

from config import ModelConfig
from models.model_utils import VanilaCnnModel
from typing import List

class BaselineModel(pl.LightningModule):
    """ 
    Makes prediction based on correlated variable trends. This native prediction is calculated by analysing the
    trend of correlated variable for latest timestep. If majority of correlated variable data is increasing 1 is predicted
    else is -1. Only give binary classificating (up/down) instead of forecasting a real value
    """

    def __init__(self) -> None:
        super().__init__()
    
    def pool_votes(self, data: List[Tensor]) -> Tensor:
        """
        Predicts up/down (1 or -1) for each data point in batch. Each data point has multiple correlated varible
        data. For e.g if a data point has 5 correlated variable and 3 of them are increasing, the prediction is 1
        if majority is decreasing, then prediction is -1
        """

        batch_size = len(data)
        out = []
        for i in range(batch_size):
            # Get deviation up/down
            up_down = torch.sign(data[i][:,-1] - data[i][:,-2])
            out.append(torch.sign(torch.mean(up_down)))
        return torch.stack(out)


    def test_step(self,  batch, batch_idx) -> None:
        x1, x2, y = batch
        y_up_down = torch.sign(y)
        y_hat = self.pool_votes(x2)
        return y_hat,y_up_down
    
    def test_epoch_end(self, outputs) -> None:
        # Combine outputs from all batches
        y_hat = torch.stack(list(map(lambda x: x[0],outputs[:-1]))).flatten()
        y = torch.stack(list(map(lambda x: x[1],outputs[:-1]))).flatten()
            
        self.log("Accuracy",torch.sum(y_hat==y)/y.shape[0])
        return outputs