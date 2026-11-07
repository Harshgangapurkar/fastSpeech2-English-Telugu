"""Minimal device-safe replacements for upstream plotting/audio-heavy utilities."""
import torch
import torch.nn.functional as F

def get_mask_from_lengths(lengths, max_len=None):
    max_len = int(lengths.max()) if max_len is None else int(max_len)
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

def pad(inputs, max_len=None):
    max_len = max(x.shape[0] for x in inputs) if max_len is None else int(max_len)
    return torch.stack([F.pad(x, (0, 0, 0, max_len-x.shape[0])) for x in inputs])
