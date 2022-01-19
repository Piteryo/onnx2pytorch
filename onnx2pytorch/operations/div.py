import torch
from torch import nn


class Div(nn.Module):
    def forward(self, input, other):
        res_type = torch.result_type(input, other)
        if not res_type.is_floating_point:
            return torch.div(input, other, rounding_mode='floor')
        true_quotient = torch.true_divide(input, other)
        res = true_quotient
        return res
