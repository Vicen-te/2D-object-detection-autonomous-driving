# types_aliases.py
import numpy as np
from typing import Tuple, List

FeatureData = Tuple[np.ndarray, List[str]] 
BBoxAbs = Tuple[float, float, float, float]                                     # (x_min, y_min, x_max, y_max)
BBoxYolo = Tuple[float, float, float, float]                                    # (x_center, y_center, width, height)
AffineMatrix = Tuple[Tuple[float, float, float], Tuple[float, float, float]]    # 2x3 matrix
