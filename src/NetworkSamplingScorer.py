
from __future__ import annotations
from collections import Counter
import copy
from joblib import Parallel, delayed
import multiprocessing as mp
import networkx as nx
import numpy as np
# import pdb
import statistics
from typing import Any, Callable, Dict, Iterable, Optional, Union
from NetworkSampling import NSMethod, NetworkSampler

def _append_to_list(li: List[Any], value: Any):
    """[summary]

    Args:
        li (Iterable[Any]): [description]
        value (Any): [description]
    """
    # li = np.append(arr=li, values=value)
    li.append(value)

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
            """
            String version of degree_sum function

            Returns:
                str: Name of degree_sum function
            """
            return 'degree_sum'

        def __repr__(self):
            """
            Representation of degree_sum function

            Returns:
                str: Representation of degree_sum function
            """
            return f'<scorer: degree_sum({graph})>'

        # pdb.set_trace(header='NetworkSamplingScorer - degree_sum - Entering degree_sum')
        degree_sum_score = np.sum(a=[deg for deg in nx.degree(G=graph)], axis=None)
        if __debug__:
            print(f'degree_sum_score: {degree_sum_score}')

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
            """
            String version of accuracy function

            Returns:
                str: Name of accuracy function
            """
            return 'accuracy'

        def __repr__(self):
            """
            Representation of accuracy function

            Returns:
                str: String representation of accuracy function
            """
            return f'<scorer: degree_sum({graph})>'

        # pdb.set_trace(header='NetworkSamplingScorer - accuracy - Entering accuracy')
        sampled_nodes = set([n for n in graph])
        target_nodes = set(targets)
        accuracy_score = len(sampled_nodes.intersection(other=target_nodes)) / float(len(target_nodes))
        if __debug__:
            print(f'accuracy_score: {accuracy_score}')

        return accuracy_score

    @staticmethod
    def distance_variance(graph: nx.Graph, start_node: int):
        """
        Computes the variance of the distribution of shortest distances from sampled nodes to the start node.

        Args:
            graph (nx.Graph): Sample of a network
            start_node (int): Start node in sample

        Returns:
            float: Distance variance
        """
        if start_node not in graph:
            raise ValueError('start_node lost during sampling')
        shortest_paths=nx.shortest_path_length(G=graph, source=start_node)
        dist_distribution = list()

        sampled_nodes = [n for n in graph]
        print(f'nodes:\n{sampled_nodes}')
        print(f'nodes length: {len(sampled_nodes)}')

        for n in sampled_nodes:
            try:
                dist_distribution.append(shortest_paths[n])
            except:
                # Very few lowest-degree nodes may become disconnected from main sample
                continue

        dist_var = statistics.variance(data=dist_distribution, xbar=statistics.mean(data=dist_distribution))

        if __debug__:
            print(f'dist_distribution: {dist_distribution}')
            print(f'dist_var: {dist_var}')

        return dist_var

    @staticmethod
    def distribution(graph: nx.Graph, transform: NSMethod=None, weights: Iterable[float]=None):
        """
        Computes the univariate frequency distribution of the nodes from a network sample after undergoing some transformation function. Optional weights are multipled to the
        product, rendering the y-value for the i^th transformed value as (transformed value [i] X frequency [i]) * weight[i]. Leaving weights to default would not commit any
        weighted scaling after applying the transformation function. Note that nodes (x-values) must be integer values.

        Args:
            graph (nx.Graph): Sample of a network
            transform (NSMethod): Function that maps node value to another numerical value before counting the frequency. Must take in node value as parameter name 'n'and graph
                                    value as parameter name 'graph'. Defaults to None.
            weights (Iterable[float], optional): Weights per node for re-scaling of frequency as the last step. Defaults to None.

        Returns:
            Iterable[Uterable[Union[int, float]]]: An iterable of iterables, with each inner (2-dim) iterable containing the transformed node and its frequency count
        """
        def __str__(self):
            """
            String version of distibution function

            Returns:
                str: Name of distribution function
            """
            return 'distribution'

        def __repr__(self):
            """
            Representation of distribution function

            Returns:
                str: String representation of distribution function
            """
            return f'<scorer: distribution({graph})>'

        # pdb.set_trace(header='NetworkSamplingScorer - distribution - Entering distribution')
        transformed_nodes = None
        if transform is None:
            transformed_nodes = list(graph.nodes)
        else:
            transform_func, transform_params = transform.func, transform.params
            transformed_nodes = [transform_func(n=n, graph=graph, **transform_params) for n in graph]
        freq_cnt = Counter(transformed_nodes)

        if __debug__:
            print(f'transformed_nodes: {transformed_nodes}')
            print(f'freq_cnt: {freq_cnt}')

        return freq_cnt
