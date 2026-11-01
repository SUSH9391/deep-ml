import numpy as np


def flip(a: np.ndarray) -> np.ndarray:
  """Reverse 1-D array a without slicing a[::-1]."""
  n = len(a)
  return a[np.arange(n - 1, -1, -1)]
  