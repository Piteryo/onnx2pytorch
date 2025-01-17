import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


class Pad(Operator):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        if input.dim() == 4 and pads.shape[0] == 8:
            input = input.squeeze(0)
            pads = [pads[3].item(), pads[7].item()]
        out = F.pad(input, pads, mode=self.mode, value=value)
        return out

    def extra_repr(self) -> str:
        return "mode={}, padding={}".format(self.mode, self.padding)
