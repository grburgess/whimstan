from dataclasses import dataclass

import numpy as np


@dataclass
class Quantile:
    low: np.ndarray
    high: np.ndarray


def extract_quantiles(quantity: np.ndarray, level: float = 90) -> Quantile:

    low = np.percentile(quantity, 50 - level * 0.5, axis=1)
    high = np.percentile(quantity, 50 + level * 0.5, axis=1)

    return Quantile(low=low, high=high)


def rank_quantile_width(quantile: Quantile) -> np.ndarray:

    width: np.ndarray = quantile.high - quantile.low

    return width / width.max()
