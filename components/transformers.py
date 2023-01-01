# Encoders and Decoders here.
# LaMDA and most GPT-style models use only a decoder, but having an encoder module in here is useful for more adaptations
import torch

from torch import nn

from .components import ChosenNorm, TransformerBlock

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
        
class Encoder(nn.Module):
    """
    Encoder model composed of a stack of transformer layers
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
        
class Decoder(nn.Module):
    """
    Decoder model composed of a stack of transformer layers
    """
    def __init__(self,
        num_decoder_layers,
        dim_model,
        vocab_size,
        chosen_pos_embedding,
        activation,
        **kwargs
    ):
        # For now, we'll mirror the LaMDA repo *exactly* until I get a hold of experimenting with other GPT-like models
        # And encoder models. This means for now encoder will be blank
        super().__init__()
        self.vocab_embedding = ChosenEmbedding(vocab_size, dim_model)
        decoder = []
        # Build decoder layers
        for _ in range(num_decoder_layers):
            decoder.append(TransformerBlock(dim_model, chosen_pos_embedding, activation, norm))
            
        # Build net
        self.decoder = nn.Sequential(*decoder)
            
    def forward(self, x):
        return self.decoder(x)
        
class GenerativeLM(nn.Module):
    """
    The full generative language model itself. Either consists of an encoder-decoder architecture or just the decoder. Outputs logits.
    """
    def __init__(self, encoder, decoder, vocab_size, dim_model, num_encoder_layers, num_decoder_layers, **kwargs):
        super().__init__()
        # Also known as the "LM head".
        # Maybe experiment with this later?
        self.projection_to_logits = nn.Sequential(
            ChosenNorm(dim_model),
            nn.Linear(dim_model, vocab_size)
        )
        
    def forward(self, x):
        raise NotImplementedError
        