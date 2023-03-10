from typing import Optional

import torch
import torch.nn as nn


def cnn_1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    dropout_prob: Optional[float] = None,
    batch_norm: bool = False,
):
    block = [nn.Conv1d(in_channels, out_channels, kernel_size), nn.ReLU()]

    # Add Dropout and Batch norm
    if dropout_prob is not None:
        if dropout_prob > 1.0 or dropout_prob < 0.0:
            raise ValueError(
                f"Dropout probability must be between 0 and 1, not {dropout_prob}"
            )
        block.append(nn.Dropout(p=dropout_prob))
    if batch_norm:
        block.append(nn.BatchNorm1d(out_channels))

    return block


def fc_block(
    input: int,
    output: int,
    dropout_prob: Optional[float] = None,
    batch_norm: bool = False,
):
    block = [nn.Linear(in_features=input, out_features=output), nn.ReLU()]

    # Add Dropout and Batch norm
    if dropout_prob is not None:
        if dropout_prob > 1.0 or dropout_prob < 0.0:
            raise ValueError(
                f"Dropout probability must be between 0 and 1, not {dropout_prob}"
            )
        block.append(nn.Dropout(p=dropout_prob))
    if batch_norm:
        block.append(nn.BatchNorm1d(output))

    return block


class VanilaCnnModel(nn.Module):
    def __init__(self, device: str = "cpu"):
        super(VanilaCnnModel, self).__init__()
        self._device = device
        self._cnn = nn.Sequential(
            *cnn_1d_block(in_channels=1, out_channels=16, kernel_size=3),
            *cnn_1d_block(in_channels=16, out_channels=32, kernel_size=3),
        ).to(self._device)

        self._fc = nn.Sequential(
            *fc_block(160, 20), *fc_block(20, 5), nn.Linear(5, 1)
        ).to(self._device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        x1 = self._cnn(x1)
        x2 = self._cnn(x2)
        x = torch.hstack([x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1)])
        x = self._fc(x)
        return x
