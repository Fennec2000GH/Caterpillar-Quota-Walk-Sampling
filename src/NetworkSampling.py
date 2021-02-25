
from __future__ import annotations
import copy
from inspect import signature
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import numpy as np
import param
import pandas as pd
import pdb
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Tuple, Union

# FuncIterable = Iterable[Tuple[Callable, Dict[str, Any]]]

# NS = Network Sampling
class NSMethod(NamedTuple):
    """Convenient container to hold any function and its parameters as a dict separately, facilitating destructuring later.

    Args:
        func (Callable): A callable that takes in at least 1 parameter
        params (Dict[str, Any]): Specific parameter values as a dict
    """
    func: Callable
    params: Dict[str, Any]

class NetworkSampler:
    """Light-weight, high-level framework used to sample networks with a given algorithm and evalutate the sample according to a given metric."""

    # METHODS
    def __init__(self, sampler: object, scorer: NSMethod) -> None:
        """Initializes the smapling algorithm and evalutation metric.

        Args:
            sampler (object): Either a sampler instance from littleballoffur library or a custom-made instance with compatible structure
            scorer (NSMethod): Function and parameters of scoring metric to evaluate network sample

        Returns:
            # None
        """
        if(not hasattr(sampler, 'sample')):
            raise ValueError('Sampler does not have a \'sample\' callable')

        self.sampler = sampler
        self.scorer = scorer
        self.prev_sample = None

    # MUTATORS
    def sample(self, graph: nx.Graph, start_node: int=None) -> nx.Graph:
        """Extracts subgraph from network by applying given sampling algorithm and parameters if provided.

        Args:
            graph (nx.Graph): Original graph or network to sample from
            start_node (int, optional): Node where sampling starts to spread. Defaults to None.

        Returns:
            nx.Graph: Sampled network.
        """
        self.prev_sample = self.sampler.sample(graph=graph, start_node=start_node)
        return self.prev_sample

    def score(self):
        """Evaluates the sample network by the provided metric. Calling the 'sample' method must be done first before calling 'score'.

        Returns:
            float: Evaluation score using given scoring metric
        """
        if self.prev_sample is None:
            raise ValueError('No calls to \'sample\' method has been made yet.')

        score_func, score_params = self.scorer.func, self.scorer.params
        score_params['graph'] = self.prev_sample
        return score_func(**score_params)

    def sample_then_score(self, graph: nx.Graph, start_node: int=None):
        """Scores network sampling algorithm with a fresh sample each time by automatically calling 'sample' first.

        Args:
            graph (nx.Graph): Original graph or network to sample from.
            start_node (int, optional): Node where sampling starts to spread. Defaults to None.

        Returns:
            Tuple[float, nx.Graph]: Tuple of score and corresponding sample of network.
        """
        self.sampler.sample(graph=graph, start_node=start_node)
        return score()


class NetworkSamplerGrid:
    """Compare and contrast different metrics of resulting samples from networks among different sampling algorithms."""
    def __init__(self, graph_group: Iterable[nx.Graph], sampler_group: Iterable[Any], scorer_group: Iterable[NSMethod]):
        """Initializes the group of sampling algorithms to compare using one or more scoring metrics.

        Args:
            graph_group (Iterable[nx.Graph]): One or more networks to sample and evaluate. Each graph having a 'name' attribute is recommended for later labeling in tables; if absent, label for a graph will be its 
                                                0-based index in graph_group.
            sampler_group (Any): One or more sampling algorithms to apply, either in the form of litleballoffur instances or a custom instance from NEtworkSamplerFunctions.py 
            scorer_group (Iterable[NSMethod]): One or more scoring metrics to apply to the resulting sampled network using each given sampling algorithm. Each value in 'params' is a fixed constant.

        Returns:
            None:
        """
        pdb.set_trace(header='NetworkSamplerGrid - __init__ - Entering initializer')

        self.graph_group = [graph for graph in graph_group]
        self.sampler_group = sampler_group
        self.scorer_group = scorer_group

        if __debug__:
            print(f'Number of graphs: {len(self.graph_group)}')
            print(f'Number of samplers: {len(self.sampler_group)}')
            print(f'Number of scorers: {len(self.scorer_group)}')

    def sample_by_graph(self, graph: nx.Graph, n_trials: int = 1, score_finalizer: str = 'mean', dir_path: str = None, n_jobs: int = 1):
        """Computes the accuracy

        Args:
            graph: Graph to sample
            n_trials (int, optional): The number of times to run the each sampling algorithm when evalutating with the same scoring metric. The end score is a single value from aggregating the scores from all the 
                                        trials. Defaults to 1.
            score_finalizer (str, optional): String indicating how to arrive at the final score for an algorithm when evaluated with a specific scoring metric. Must be either 'mean' or 'median'. Defaults to 'mean'.
            dir_path (str, optional): Path for directory holdin the pickled results of NetworkSamplerGrid. The extension at the end must be '.pkl'. If None, nothing will be stored in memory. Defaults to None.
            n_jobs (int, optional): Number of jobs to divide sampling across all graphs with multiprocessing. Defaults to 1.

        Returns:
            Dict[pd.Dataframe]: A ditionary of Dataframes, each corresponding to a score report for the same scoring metric across all given graphs and network sampling algorithms. Each Dataframe applies distinct 
                                network samples as rows and sampling algorithm names as columns.
        """

        pdb.set_trace(header='NetworkSamplerGrid - samply_by_graph - Entering function')
        col_labels = [str(scorer.func) for scorer in self.scorer_group]
        retval = pd.DataFrame(columns=col_labels)

        pdb.set_trace(header='NetworkSamplerGrid - sample_by_graph - Sampling chose graph with each provided smapling algorithm')
        for sampler in self.sampler_group:
            if __debug__:
                print(f'sampler: {sampler}')
            row = list()
            for scorer in self.scorer_group:
                # Computing score / distribution
                ns = NetworkSampler(sampler=sampler, scorer=scorer)
                sample = ns.sample(graph=graph, start_node=...)
                score_result = ns.scoree()
                row.append(score_result)

                # Drawing and saving sample
                nx.draw(G=sample, pos=nx.spring_layout(G=sample))
                plt.show(block=False)
                plt.savefig(f'./Graph Plots/{str(sampler)}.jpg', format='JPG')

            # Appends to dataframe new row of scores computed via different metrics in score_group
            retval.append(other=pd.Series(data=row, index=col_labels))

        return retval


    def sample_stored_graphs(self, n_trials: Iterables[int]=None, score_finalizer: str = 'mean', dir_path: str = None, n_jobs: int = 1):
        """
        Samples each given network during initialization and creates a dataframe with sampling algorithm as row and scoring metric as column. The collection of such dataframes
        are stored into a dict, keyed by network name if it exists or the index of the network during class initialization.

        Args:
            n_trials (Iterable[int], optional): [description]. Defaults to None.
            score_finalizer (str, optional): [description]. Defaults to 'mean'.
            dir_path (str, optional): [description]. Defaults to None.
            n_jobs (int, optional): [description]. Defaults to 1.

        Returns:
            dict[pd.DataFrame]: Dictionary ofdataframes, each being a scoreboard across all given sampling algorithms measured with all given scoring metrics for each network
        """
        retval = dict()
        for index, graph in enumerate(self.graph_group):
            retval[graph['name'] if 'name' in graph else str(index)] = sample_by_graph(graph=graph, n_trials=n_trials[index], score_finalizer=score_finalizer, n_jobs=n_jobs)
        return retval
