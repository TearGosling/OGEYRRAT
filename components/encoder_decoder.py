# Encoders and Decoders here.
# LaMDA and most GPT-style models use only a decoder, but having an encoder module in here is useful for more adaptations
import torch

from torch import nn

class EncoderLayer(nn.Module):
    """
    Layer for an encoder.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
        

class DecoderLayer(nn.Module):
    """
    Layer for a decoder.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
class Encoder(nn.Module):
    """
    Encoder model composed of a stack of encoder layers
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
        
class Decoder(nn.Module):
    """
    Decoder model composed of a stack of decoder layers
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError