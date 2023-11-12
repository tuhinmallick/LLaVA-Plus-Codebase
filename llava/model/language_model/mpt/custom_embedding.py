import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SharedEmbedding(nn.Embedding):

    def forward(self, input: Tensor, unembed: bool=False) -> Tensor:
        return F.linear(input, self.weight) if unembed else super().forward(input)