import re
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
from nptyping import Float, NDArray
from torch import Tensor
import torch

@dataclass
class VariableMetadata:
    """Stores the columns number of the variable, and column numbers of correlated variables"""

    var_idx: int
    corr_var_idxs: List[int]


@dataclass
class VariableData:
    """
    Stores an instance of data point, time series data of main variable and correlated variables with correlated
    variables looking one timestep in future. Contains the ground truth prediction of main variable for next timestep
    """

    x1: NDArray[Any, Float]  # main variable
    x2: NDArray[Any, Float] # correlated variables
    y: Float

    def __post_init__(self):
        conditions = [
            np.isfinite(self.x1).all(),
            np.isfinite(self.x2).all(),
            np.isfinite(self.y),
            self.x2.shape[1] == self.x1.shape[1] + 1,
        ]
        if not all(conditions):
            raise ValueError("Values contain NaN element or there is a shape mismatch")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes data to remove all symbols and convert the strings into numbers. Removes irrelevant
    columns like dates
    """
    if "Month" in df.columns:
        df = df.drop(columns="Month")
    df = df.rename(columns=lambda x: re.sub("\n", "", x))
    df = df.replace("[K,B,M,T,\\%,\\n,\\n%,^\\s+,^\\s$]", "", regex=True)
    df = df.apply(pd.to_numeric)
    return df


def find_correlated_indices(
    df: pd.DataFrame, min_period: int = 10
) -> List[VariableMetadata]:
    """Calculates correlation between variables and returns the information about the
    indices of correlated variables with the main variables

    :param df: time series data of all variables
    :param min_period: min. finite values to calculate correlation
    """
    correlations = df.corr(method="pearson", min_periods=min_period).values

    # Keep only upper trianglular as correlation matrix is symmetric, also reject diagonal entries(self correlation)
    correlations = np.triu(correlations, k=1)

    # Find indices of high correlation
    high_corr_idxs = np.where(correlations > 0.8)
    variable_metadatas = []
    for var_idx in set(high_corr_idxs[0]):
        corr_var_idxs = high_corr_idxs[1][np.where(high_corr_idxs[0] == var_idx)]
        variable_metadatas.append(
            VariableMetadata(var_idx=var_idx, corr_var_idxs=corr_var_idxs)
        )
    return variable_metadatas


def create_rolling_data(
    df: pd.DataFrame, var_metadatas: List[VariableMetadata], x_len
) -> List[VariableData]:
    """Returns all datapoints in a moving window fashion going over all variables and its correlated variables.

    :param df: processed dataframe, contains only variable columns with numbers
    :param var_metadatas: information about variable and its correlated variables indices
    :param x_len: length of past information used to next timestep prediction
    """
    data = []
    len_xy = x_len + 1  # including prediction value, (length of corr var data)

    for var_metadata in var_metadatas:
        variable = df.iloc[:, var_metadata.var_idx].values
        for i in range(variable.shape[0] - (x_len + 1)):

            # Reject cases with nan values
            if np.isfinite(variable[i : i + len_xy]).all():
                x1 = variable[i : i + x_len]
                y = variable[i + x_len]
            else:
                continue

            # Add all correlated variables series
            x2 = []
            for corr_var_idx in var_metadata.corr_var_idxs:
                correlated_variable = df.iloc[:, corr_var_idx].values
                if all(np.isfinite(correlated_variable[i : i + len_xy])):
                    x2.append(correlated_variable[i : i + len_xy])

            if len(x2) > 0:
                data.append(
                    VariableData(np.asarray(x1).reshape(1, -1), np.asarray(x2), y)
                )

    return data


def collate(batch) -> Tuple[Tensor]:
    x1 = torch.stack(list(map(lambda x: x[0],batch)))
    x2 = list(map(lambda x: x[1],batch))
    y = torch.stack(list(map(lambda x: x[2],batch)))
    
    return (x1,x2,y)