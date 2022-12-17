import torch
import torch.nn.functional as F

from torch import nn

# Here is a giant dictionary of different activation functions you can use. The idea is you can just specify a type of activation in a .yaml config, and it'll pick out the correct activation from this.
# It's messy, but it works.
BIG_ASS_ACTIVATION_DICTIONARY = {
    "geglu": GatedLinearUnit(nn.GELU()),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "swiglu": GatedLinearUnit(nn.SiLU()),
    "swish": nn.SiLU(),
    "squaredrelu": SquareRelu(),
}

class GatedLinearUnit(nn.Module):
    """
    A gated linear unit. You can plug any function into this to create a variant of the GLU such as GEGLU with GELU or SwiGLU with Swish.
    """
    def __init__(self, glu_fn=nn.Identity()):
        self.glu_fn = glu_fn
        
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * self.glu_fn(gates)
        
class SquaredRelu(nn.Module):
    """
    The Squared ReLU activation function, that is, the output of ReLU, squared.
    """
    def forward(self, x):
        return F.relu(x) ** 2