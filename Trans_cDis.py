import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import pywt
import numpy as np

#from run_trans import batch_size
from millet.model.pooling import MILConjunctivePooling


class CNNTransformerDiscriminator(nn.Module):
    def __init__(self, seq_len=150, in_channels=16, cnn_out_channels=64, emb_dim=128, num_heads=8, depth=3,
                 forward_drop_rate=0.3, attn_drop_rate=0.3, num_classes=21):
        super(CNNTransformerDiscriminator, self).__init__()
        # CNN-based feature extractor (set to accept `in_channels` as input)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_out_channels),  # Add Batch Normalization here
            nn.LeakyReLU(0.2),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_out_channels),  # Add Batch Normalization here
            nn.LeakyReLU(0.2),
            nn.Conv1d(cnn_out_channels, num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_classes),  # Add Batch Normalization here
            nn.LeakyReLU(0.2),
        )

        # Embedding layer for conditional labels
        self.label_embedding = nn.Embedding(num_classes, seq_len)

        # Transformer encoder layers
        self.transformer = TransformerEncoder(
            depth=depth,
            emb_size=emb_dim,
            num_heads=num_heads,
            drop_p=attn_drop_rate,
            forward_drop_p=forward_drop_rate
        )

        # Classification head
        self.cls_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x, labels):
        # Ensure x has three dimensions and the expected channel size
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Embed the labels
        label_embed = self.label_embedding(labels).unsqueeze(1)  # [batch_size, 1, seq_len]
        # Expand and add the label embeddings to match the input
        label_embed = label_embed.expand(-1, x.size(1), -1)  # Broadcast to match dimensions
        pool_len = label_embed.size(2)
        x = F.adaptive_avg_pool1d(x, pool_len)

        # Concatenate along the channel/feature dimension
        x = torch.cat([x, label_embed], dim=1)  # [batch_size, original_channels + 1, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, cnn_out_channels]
        # Pass through CNN
        x = self.cnn(x)  # [batch_size, cnn_out_channels, seq_len]

        # Permute for Transformer compatibility

        # Pass through Transformer
        #x = self.transformer(x)  # [batch_size, seq_len, emb_dim]

        # Pool and classify
        x = x.mean(dim=2)  # Global average pooling [batch_size, emb_dim]
        out = x
        #out = self.cls_head(x)
        return out

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


class TransformerEncoder(nn.Module):
    def __init__(self, depth=3, emb_size=128, num_heads=8, drop_p=0.3, forward_drop_p=0.3):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, drop_p, forward_drop_p=forward_drop_p)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=8, drop_p=0.1, forward_expansion=4, forward_drop_p=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.feed_forward = FeedForwardBlock(emb_size, forward_expansion, forward_drop_p)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
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
        # Reshape for multi-head attention
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # Scaled Dot-Product Attention
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            energy.masked_fill_(~mask, float("-inf"))

        scaling = math.sqrt(self.emb_size // self.num_heads)
        att = F.softmax(energy / scaling)
        att = self.att_drop(att)
        # Aggregate values and project
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