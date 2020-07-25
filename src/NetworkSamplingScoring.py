import numpy as np
from typing import Callable


def zero_one_loss(target_nodes: np.ndarray, test_nodes: np.ndarray) -> float:
    """
    Scoring function used to gauge accuracy of NSModel

    Parameters
    target_nodes - expected set of nodes to be visited
    test_nodes - nodes visited by some network sampling function to be scored
    """
    try:
        if type(target_nodes) != np.ndarray or type(test_nodes) != np.ndarray:
            raise TypeError("Both inputs must be numpy arrays")
        if len(target_nodes.shape) != 1 or len(test_nodes.shape) != 1:
            raise ValueError("Both inputs must be 1-dimensional")
        if target_nodes.size == 0:
            raise ValueError("Target nodes cannot be empty")
    except (TypeError, ValueError) as error:
        print(str(error))

    return len(set(target_nodes).intersection(set(test_nodes))) / len(target_nodes)
