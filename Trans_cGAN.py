import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import pywt
import numpy as np
from torch.onnx.symbolic_opset9 import unsqueeze
from millet.model.pooling import MILConjunctivePooling
#from run_Trans_cGAN import num_classes


class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes=21):
        super(ConditionalBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        # Initialize embedding layers
        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.beta_embed.weight)
    def forward(self, x, labels):
        out = self.bn(x)
        gamma = self.gamma_embed(labels)  # [batch_size, num_features, 1]
        beta = self.beta_embed(labels)    # [batch_size, num_features, 1]
        if len(out.shape) == 3:  # In case `out` is [batch_size, num_features, seq_len]
           gamma = gamma.expand(-1, -1, out.size(2)) # Shape: [batch_size, num_features, seq_len]
           beta = beta.expand(-1, -1, out.size(2))
        out = gamma * out + beta
        return out

class AdvancedDopplerGenerator(nn.Module):
    def __init__(self, seq_len=150, latent_dim=100, embed_dim=16, depth=4, num_heads=8,
                 forward_drop_rate=0.1, attn_drop_rate=0.1, doppler_layers=3, output_channels=16, num_classes=10):
        super(AdvancedDopplerGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Embedding layer for conditional labels
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        # Linear layer to map latent space to initial sequence
        self.l1 = nn.Linear(self.latent_dim + embed_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.layer_norm = ConditionalBatchNorm1d(self.embed_dim, num_classes)
        # Transformer encoder blocks
        self.transformer = TransformerEncoder(
            depth=depth,
            emb_size=embed_dim,
            num_heads=num_heads,
            drop_p=attn_drop_rate,
            forward_drop_p=forward_drop_rate,
            num_classes=num_classes
        )
        self.attention_pooling = MILConjunctivePooling(d_in=embed_dim, n_clz=num_heads, dropout=0.2,
                                                       apply_positional_encoding=True)
        self.proj_to_embed_dim = nn.Linear(self.num_heads, self.embed_dim)
        # Output layer to adjust embedding to match the required output channels
        self.out_layer = nn.Linear(embed_dim, output_channels)
        self.doppler_layers = nn.ModuleList([DopplerEffectLayer(seq_len, embed_dim) for _ in range(doppler_layers)])


    def forward(self, z, labels, speed=None, angle=None, distance=None):
        # Embed the labels
        label_embed = self.label_embedding(labels)
        # Concatenate latent vector with label embedding
        z = torch.cat([z, label_embed], dim=1)
        # Map latent vector to initial sequence embedding
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        # Add positional embedding
        x = self.pos_embed + x
        # Pass through Transformer blocks
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        pooling_output = self.attention_pooling(x)
        x = pooling_output['bag_logits']
        x = self.proj_to_embed_dim(x)
        x = self.layer_norm(x, labels)
        x = x.unsqueeze(2)
        x = x.permute(0, 2, 1)
        for doppler_layer in self.doppler_layers:
            x = doppler_layer(x, speed=speed, angle=angle, distance=distance)
        #x = self.apply_wavelet_denoising(x)
        # Project to final signal output and reshape for discriminator compatibility
        x = self.out_layer(x).permute(0, 2, 1)  # Shape: [batch_size, output_channels, seq_len]
        return x

    def apply_wavelet_denoising(self, x, wavelet="db1", level=1):
        """
        Apply wavelet denoising to each sample in the batch.
        """
        x_denoised = []
        for i in range(x.size(0)):  # Loop over batch size
            denoised_signal = self.wavelet_denoise(x[i].cpu().detach().numpy(), wavelet, level)
            x_denoised.append(torch.tensor(denoised_signal, device=x.device))
        return torch.stack(x_denoised)

    def wavelet_denoise(self, signal, wavelet="db1", level=1):
        """
        Denoise using wavelet decomposition and thresholding.
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        threshold = np.sqrt(2 * np.log(len(signal))) * np.median(np.abs(coeffs[-1])) / 0.6745
        denoised_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
        return pywt.waverec(denoised_coeffs, wavelet)

class DopplerEffectLayer(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(DopplerEffectLayer, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, x, speed=None, angle=None, distance=None):
        """
        Apply Doppler shift based on speed, angle, and distance.
        Simulates frequency shifts due to UAV kinematic parameters.
        """
        if speed is None or angle is None or distance is None:
            return x  # No Doppler shift applied if parameters are missing
        # Frequency shift factor, based on simplified Doppler shift formula
        doppler_shift = self.calculate_doppler_shift(speed, angle, distance)
        # Apply Doppler shift to the sequence embeddings
        doppler_effect = torch.sin(
            torch.arange(0, self.seq_len, device=x.device).float().unsqueeze(0) * doppler_shift.unsqueeze(1))
        doppler_effect = doppler_effect.clamp(min=-1, max=1)
        doppler_effect = doppler_effect.unsqueeze(-1).permute(1, 0, 2)
        doppler_effect = doppler_effect.permute(1, 0, 2)
        if torch.isnan(doppler_effect).any():
            print('NaN detached in doppler effect')
            return x
        x = x * doppler_effect
        return x

    def calculate_doppler_shift(self, speed, angle, distance):
        """
        Calculate Doppler shift as a function of speed, angle, and distance.
        Using a simplified model based on UAV parameters.
        """
        c = 3e8  # Speed of light in m/s
        base_freq = 2.4e9  # Base frequency in Hz (for example, WiFi or control signals)
        # Doppler shift formula: f_d = (v * f_c * cos(theta)) / c
        shift = (speed * base_freq * torch.cos(angle)) / c
        return shift / (distance + 1e-6)  # Avoid division by zero

class TransformerEncoder(nn.Module):
    def __init__(self, depth=4, emb_size=16, num_heads=8, drop_p=0.1, forward_drop_p=0.1, num_classes=21):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, drop_p, forward_drop_p=forward_drop_p, num_classes=num_classes)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=8, drop_p=0.2, forward_expansion=4, forward_drop_p=0.2, num_classes=21):
        super(TransformerEncoderBlock, self).__init__()
        self.num_classes = num_classes
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.feed_forward = FeedForwardBlock(emb_size, forward_expansion, forward_drop_p)
        self.norm1 = nn.BatchNorm1d(emb_size)
        self.norm2 = nn.BatchNorm1d(emb_size)
        self.dropout1 = nn.Dropout(drop_p)
        self.dropout2 = nn.Dropout(drop_p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = x.permute(0, 2, 1)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, float("-inf"))
        scaling = math.sqrt(self.emb_size // self.num_heads)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )