
import numpy as np

def zero_one_loss(target_nodes: np.ndarray, test_nodes: np.ndarray) -> float:
    """
    Scoring function used to gauge accuracy of NSModel

    Parameters
    :param target_nodes: Expected set of nodes to be visited
    :param test_nodes: Nodes visited by some network sampling function to be scored
    :return: Proportion of nodes in target_nodes that are actually visited in test_nodes
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
    true_positives = set(target_nodes).intersection(set(test_nodes))
    return len(true_positives) / len(set(target_nodes))

def zero_one_loss_fpp(target_nodes: np.ndarray, test_nodes: np.ndarray, penalty: float = 1.0) -> float:
    """
    Stands for zero_one_loss false positive penalty. For every false positive, that is, node in test_nodes
    but not in expected target_nodes, the final score is lessened by the penalty value.

    :param target_nodes: Expected set of nodes to be visited
    :param test_nodes: Nodes visited by some network sampling function to be scored
    :param penalty: Number to be subtracted from score in regular zero_one_loss for every false positive node in test_nodes
    :return: Proportion of nodes in target_nodes that are actually visited in test_nodes with penalties accounted for
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
    false_positives = set(test_nodes).difference(set(target_nodes))
    return zero_one_loss(target_nodes=target_nodes, test_nodes=test_nodes) - penalty * len(false_positives)
