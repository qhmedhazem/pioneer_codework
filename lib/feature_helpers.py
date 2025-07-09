import math
from typing import List, Tuple


def distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """
    Calculates Euclidean distance between two 2D points.
    """
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    return math.sqrt(dx * dx + dy * dy)


def reduce_vector(vec, status):
    return [v for v, s in zip(vec, status) if s]
