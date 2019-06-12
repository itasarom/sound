
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CLNNModule(nn.Module):
    def __init__(self, in_channels, out_channels, window_size, stride=1, dilation=1, bias=True):
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size

        self.convolution = nn.Conv1d(in_channels, out_channels, 
            kernel_size=2*window_size + 1, dilation=dilation, 
            stride=1, padding=window_size, padding_mode="zeros")


    def forward(self, input):
        batch_size, max_time, n_features = input.shape

        assert self.in_channels == n_features, (self.in_channels, n_features)

        input = input.permute(0, 2, 1)

        result = self.convolution(input)

        result = result.permute(0, 2, 1)

        return result


# class TemporalPool(nn.Module):
#     def __init__(self, base_function):
#         super().__init__()
#         self.base_function = base_function

#     def forward(self, input):


#         result = self.base_function(input, dim=1)
#         # if hasattr(result, "values"): 
#             # result = result.values()

#         return result

class TemporalPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        result = self.pool(input)
        result = result.permute(0, 2, 1)

        return result


class BatchNorm(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)


    def forward(self, input):
        input = input.permute(0, 2, 1)
        result = self.bn(input)
        result = result.permute(0, 2, 1)

        return result


class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.shape[0]

        result = input.contiguous().reshape(batch_size, -1)

        return result






        





