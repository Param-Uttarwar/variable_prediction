import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from data_ops.data_utils import (VariableData, create_rolling_data,
                                 find_correlated_indices, process_data)

logger = logging.getLogger(__name__)


class VariableDataset(Dataset):
    def __init__(self, device: str, x_len: int = 6) -> None:
        """Dataset class

        :param x_len: length of past information used to next timestep prediction
        """
        self._x_len = x_len
        self._device = device

    def load_from_csv(self, path: Path) -> None:
        """
        Reads time series variable data from csv file. Variables are arranged as columns. The data is converted
        to (X1,X2,y) that are generated in a moving window fashion. y is the next timestep prediction and X1 is the
        variable values for past timesteps, X2 is the correlated variable data for past and next timestep
        """

        df = pd.read_csv(path)
        df = process_data(df)
        var_metadatas = find_correlated_indices(df)
        data = create_rolling_data(df, var_metadatas, self._x_len)
        self._data = data
        logging.info(f"Data parsed and loaded of length {len(data)}")

    def __len__(self) -> float:
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        datapoint: VariableData = self._data[index]

        # Reduce mean
        x1_mean, x1_std = datapoint.x1.mean(), datapoint.x1.std()
        x2_mean, x2_std = datapoint.x2.mean(axis=1), datapoint.x2.std(axis=1)

        x2_std[np.where(x2_std < 0.1)] = 1
        x1_std = 1 if x1_std < 0.1 else x1_std

        x1 = (datapoint.x1 - x1_mean) / x1_std
        x2 = (datapoint.x2 - x2_mean[:, None]) / x2_std[:, None]
        y = (datapoint.y - x1_mean) / x1_std

        return (
            Tensor(x1).to(self._device),
            Tensor(x2[0, None, :]).to(self._device),
            Tensor([y]).to(self._device),
        )
