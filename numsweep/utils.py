"""General utilities."""


import torch
import numpy as np


def truncate_significand(t, n):
    if t.dtype == torch.float32:
        a = np.array(t)
        b = np.frombuffer(a.tobytes(), dtype=np.int32)
        b = b & (-1 << n)
        a = np.frombuffer(b.tobytes(), dtype=np.float32).reshape(a.shape)
    else:
        a = np.array(t)
        b = np.frombuffer(a.tobytes(), dtype=np.int64)
        b = b & (-1 << n)
        a = np.frombuffer(b.tobytes(), dtype=np.float64).reshape(a.shape)
    return torch.tensor(a)
