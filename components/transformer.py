# Everything will be in one file for now.
# When things start becoming very, very long, that's when we start splitting files off.

import torch

from torch import nn

from .activations import BIG_ASS_ACTIVATION_DICTIONARY as BAAD
from .position_embeddings import POS_EMBED_DICT

class Residual(nn.Module):
    """
    The residual. Applies a function to input x, and then re-adds the original x.
    TODO: Add residual scaling as implemented in x-transformers.
    """
    def __init__(self, fn):
        super().__init___()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ChosenActivation(nn.Module):
    """
    This is a wrapper for a chosen activation function that is in the Big-Ass Activation Dictionary (BAAD).
    """
    def __init__(self, activation: str):
        super().__init__
        self.activation = BAAD[activation]
        
    def forward(self, x):
        return self.activation(x)
 

class ChosenEmbedding(nn.Module):
    """
    Embedding. For now, it'll just be standard nn.Embedding, but having this as its own class allows for future experiments.
    Experimenting is the whole purpose of this repo, after all.
    """
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, x):
        return self.embed(x)
        

class ChosenNorm(nn.Module):
    """
    Wrapper to normalize an input. For now, it'll just be standard nn.LayerNorm, but stuff like RMS norm should be implemented soon.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x)
                
class PrepareQKV(nn.Module):
    """
    Inspired by nn.labml.ai's annotation of the Transformer model, there will be a separate class for the purposes of translating the input into Q, K, and V.
    Experimentation can be done here too.
    """
    def __init__(self):
        super().__init__()
    
    def forward(x):
        raise NotImplementedError
        
class Attention(nn.Module):
    """
    Multi-head attention. Lots of fun here.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
class FeedForwardNetwork(nn.Module):
    """
    Feedforward network. Allows for plenty of customization... later.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        raise NotImplementedError