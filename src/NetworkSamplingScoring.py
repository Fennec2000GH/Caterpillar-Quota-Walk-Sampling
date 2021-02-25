
from collections import Counter
import copy
from joblib import Parallel, delayed
from inspect import signature
import multiprocessing as mp
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict, Iterable, Optional, Union

from NetworkSampling import NSMethod, NetworkSampler

class NetworkSamplingScorer:
    @staticmethod
    def degree_sum(graph: nx.Graph):
        """
        Sums up the degree of all nodes in given netork, assumed to be a sample of a larger network. Note that this metric is only useful when comparing samples with the same size
        (number of nodes) sampled from the same network.

        Args:
            graph (nx.Graph): Sample of a network

        Returns:
            int: Sum of degrees for all nodes in sample
        """
        def __str__(self):
            """String version of function name"""
            return 'degree_sum'

        def __repr__(self):
            """
            Representation of degree_sum function

            Returns:
                str: String representation of degree_sum function
            """
            return f'<scorer: degree_sum({graph})>'

        degree_sum_score = np.sum(a=[deg for deg in nx.degree(G=graph)], axis=None)
        return degree_sum_score

    @staticmethod
    def accuracy(graph: nx.Graph, targets: Iterable[int]):
        """
        Computes the fraction of target nodes that have been visited in the network sample. Note that this metric is only useful when comparing samples with the same size
        (number of nodes) sampled from the same network.

        Args:
            graph (nx.Graph): Sample of a network

        Returns:
            float: The proportion of target nodes visited to the number of target nodes
        """
        def __str__(self):
            """String version sfd
            """
        def __repr__(self):
            """
            Representation of accuracy function

            Returns:
                str: String representation of accuracy function
            """
            return f'<scorer: degree_sum({graph})>'

        if __debug__:
            print(f'graph type: {type(graph)}')
        sampled_nodes = set([n for n in graph])
        target_nodes = set(targets)
        accuracy_score = len(sampled_nodes.intersection(other=target_nodes)) / float(len(target_nodes))
        return accuracy_score

    @staticmethod
    def distribution(graph: nx.Graph, transform: Callable=lambda x : x, weights: Iterable[float]=None):
        """
        Computes the univariate frequency distribution of the nodes from a network sample after undergoing some transformation function. Optional weights are multipled to the
        product, rendering the y-value for the i^th transformed value as (transformed value [i] X frequency [i]) * weight[i]. Leaving weights to default would not commit any
        weighted scaling after applying the transformation function. Note that nodes (x-values) must be integer values.

        Args:
            graph (nx.Graph): Sample of a network
            transform (Callable): Function that maps node value to another numerical value before counting the frequency. Must take in node value as the first parameter. Defaults to no change.
            weights (Iterable[float], optional): Weights per node for re-scaling of frequency as the last step. Defaults to None.

        Returns:
            Iterable[Uterable[Union[int, float]]]: An iterable of iterables, with each inner (2-dim) iterable containing the transformed node and its frequency count
        """
        transformed_nodes = [transform(n) for n in graph]
        freq_cnt = Counter(transformed_nodes)
        freq_dist = np.empty(shape=(2,), dtype=(Union[int, float], int))
        update = lambda kv_pair : (freq_dist := np.append(arr=freq_dist, values=kv_pair))
        with Parallel(n_jobs=mp.cpu_count(), backend='multiprocessing') as parallel:
            parallel(delayed(function=update)(np.asarray(a=[k, v])) for k, v in freq_cnt.items())
        return freq_dist
