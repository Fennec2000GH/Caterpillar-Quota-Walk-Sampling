
import copy
import heapq
from joblib.parallel import Parallel, delayed
import multiprocessing as mp
import networkx as nx
import numpy as np
import pdb
import random
from typing import Any, Callable, Dict, Iterable, Union
from NetworkSampling import NSMethod

# PROPOSED METHOD
#region
class CaterpillarQuotaWalkSampler:
    """
    Proposed algorithm takes in two percentage quotas (q1, q2 such that q1 < q2 <= 100%). The
    minimal quorum of weighted neighbors, when ranked from highest to lowest weight, that
    cumulatively meets or exceeds Q1 continue to extend the central path(s) of the sampled
    subgraph. Analogously, the minimal quorum of weighted neighbors that cumulatively meet or
    exceed Q2, but minus the neighbors already included in Q1, extend as single-edge branches
    of the sampled subgraph based on the caterpillar tree graph model.
    """
    def __init__(self, number_of_nodes: int=100, q1: float=0.01, q2: float=0.05):
        """
        Initializes metadata before smapling occurs

        Args:
            number_of_nodes (int, optional): The number of nodes to sample before stopping. Defaults to 100.
            q1 (float, optional): The proportion of top-weighted neighboring nodes to visit and extend into new caterpillar graphs
            q2 (float, optional): The proportion of top-weighted neighboring nodes not already covered in Q1 to visit once and become dead end (no deeper traversal allowed)
        """

        # pdb.set_trace(header='CaterpillarQuotaWalk - __init__ - Initializing sampler')
        self.number_of_nodes = number_of_nodes
        self.q1 = q1
        self.q2 = q2

        if __debug__:
            print(f'number_of_nodes: {self.number_of_nodes}')
            print(f'q1: {self.q1}')

    #region
    def sample(self, graph: nx.Graph, start_node: int=None):
        """
        Samples the network

        Args:
            graph (nx.Graph): Network to be sampled
            start_node (int, optional): Node to start the sampling. Defaults to None.
        """
        # pdb.set_trace(header='CaterpillarQuotaWalk - sample - Entering sample')
        if __debug__:
            print('graph: {graph}')
            print('start_node: {start_node}')

        # Node collections
        visited = set([start_node])
        curr_layer = set([start_node])
        next_layer = set()
        # stop = False  # Indicates whether to stop loops once desired number of nodes is sampled

        # DEBUGGING PURPOSES
        layer_counter = 0
        node_counter = 1

        # Choosing nodes to contribute to sampling
        def _sample_at_node(n: Any):
            """
            Internal helper function that carries out the actual algorithm at a specific visited node

            Args:
                n (Any): Node to visit neighbors from

            Raises:
                ValueError: n is not part of the graph's vertex set
                ValueError: n is already visited
            """
            # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - Entering _sample_at_node')
            if __debug__:
                print(f'n: {n}')

            if n not in graph:
                raise ValueError(f'Node {n} must exist inside network {graph}')
            if n not in visited:
                raise ValueError(f'Node {n} must be already visited.')

            # unvisited neighboring nodes of n rnaked by degree descending
            unvisited_nbrs = []
            for nbr in graph.neighbors(n=n):
                if nbr not in visited:
                    unvisited_nbrs.append(nbr)
            degree_ranked_desc_nbrs = np.asarray(a=sorted(unvisited_nbrs, key=lambda x : graph.degree[x], reverse=True))

            # No unvisited neighbors left
            if degree_ranked_desc_nbrs.size == 0:
                return

            cumsum_degree = np.cumsum(a=[graph.degree[nbr] for nbr in degree_ranked_desc_nbrs], axis=None)
            sum_degree = np.sum(a=[graph.degree[nbr] for nbr in degree_ranked_desc_nbrs], axis=None)

            # pdb.set_trace(header='CaterpillarQuotaWalk - sample - sample_at_node - Computing quota weights and indexes')
            # Computing weight threshold (ceiling) for unvisited neighbors given q1, q2
            q1_quota_weight, q2_quota_weight = self.q1 * sum_degree, self.q2 * sum_degree

            # Computing indexes of the successor of the last neigbor belonging under q1 threshold and q2 threshold, respectively
            q1_index, q2_index = np.argwhere(a=cumsum_degree > q1_quota_weight)[0], np.argwhere(a=cumsum_degree > q2_quota_weight)[0]

            # No nodes have degree that falls at or under even the q1 threshhold, so manually designate first node (index is 0) to belong to q1 group
            q1_index = max([q1_index, 1])

            if __debug__:
                print(f'degree_ranked_desc_nbrs:\n{degree_ranked_desc_nbrs}')
                print(f'cumsum_degree:\n{cumsum_degree}')
                print(f'sum_degree:\n{sum_degree}')

            # Visiting new nodes
            # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - Adding new nodes to visited and swapping current layer with new layer')
            nonlocal curr_layer
            nonlocal next_layer
            nonlocal node_counter
            for nbr in degree_ranked_desc_nbrs[0:int(q1_index)]:
                visited.add(nbr)
                next_layer.add(nbr)

                node_counter += 1
                if __debug__:
                    print(f'node_counter: {node_counter}')

            # Brief check for exceeding the node limit for sampling
            if node_counter > self.number_of_nodes:
                return

            for nbr in degree_ranked_desc_nbrs[int(q1_index):int(q2_index)]:
                visited.add(nbr)

                node_counter += 1
                if __debug__:
                    print(f'node_counter: {node_counter}')

            # Final check for exceeding the node limit for sampling
            if node_counter > self.number_of_nodes:
                return

        # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - Calling sampling algorithm for each node in current layer')

        while node_counter < self.number_of_nodes:
            layer_counter += 1
            if __debug__:
                print(f'layer_counter: {layer_counter}')

            for n in set(visited):
                _sample_at_node(n=n)

        visited_list = sorted(list(visited), key=lambda n: graph.degree[n], reverse=True)[:self.number_of_nodes]

        # Forces start_node to be part of visited set
        if start_node not in visited_list:
            visited_list[-1] = start_node

        sampled_network = graph.subgraph(nodes=visited_list)

        if __debug__:
            print(f'visited_list: {visited_list}')
            print(f'sampled_network: {sampled_network}')

        # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - returning sampled network')
        return sampled_network
#endregion
#endregion

#region
class CaterpillarQuotaBFSSampler:
    """
    Variant of the CaterpillarQuotaWalkSampler but takes in only 1 quota proportion, q1. Rather than iterate through a layer of visited nodes
    during each iteration, all unvisited nodes adjacent to any visited node are grouped together as a whole set. The q1 proportion of nodes with
    the highest degrees are visited, and then all adjacent unvisited nodes previously unaccounted for are added to the unvsisited set.
    """
    def __init__(self, number_of_nodes: int=100, q1: float=0.01):
        """
        Initializes metadata before smapling occurs

        Args:
            number_of_nodes (int, optional): The number of nodes to sample before stopping. Defaults to 100.
            q1 (float, optional): The proportion of top-weighted neighboring nodes to visit and extend into new caterpillar graphs
        """

        # pdb.set_trace(header='CaterpillarQuotaWalk - __init__ - Initializing sampler')
        self.number_of_nodes = number_of_nodes
        self.q1 = q1

        if __debug__:
            print(f'number_of_nodes: {self.number_of_nodes}')
            print(f'q1: {self.q1}')

#region
    def sample(self, graph: nx.Graph, start_node: int=None):
        """
        Samples the network

        Args:
            graph (nx.Graph): Network to be sampled
            start_node (int, optional): Node to start the sampling. Defaults to None.
        """
        # pdb.set_trace(header='CaterpillarQuotaWalk - sample - Entering sample')
        if __debug__:
            print('graph: {graph}')
            print('start_node: {start_node}')

        # Node collections
        visited = set([start_node])
        unvisited_nodes = [(-graph.degree[nbr], nbr) for nbr in graph.neighbors(n=start_node)]  # priority queue of 2-elemnt tuple (-degree, node id)

        # DEBUGGING PURPOSES
        layer_counter = 0
        node_counter = 1

        # Choosing nodes to contribute to sampling
        def _sample_at_node(n: Any):
            """
            Internal helper function that carries out the actual algorithm at a specific visited node

            Args:
                n (Any): Node to visit neighbors from

            Raises:
                ValueError: n is not part of the graph's vertex set
                ValueError: n is already visited
            """
            # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - Entering _sample_at_node')
            if __debug__:
                print(f'n: {n}')
            elif n not in graph:
                raise ValueError(f'Node {n} must exist inside network {graph}')

            # unvisited neighboring nodes of n rnaked by degree descending
            for nbr in graph.neighbors(n=n):
                if nbr not in visited:
                    heapq.heappush(unvisited_nodes, (-graph.degree[nbr], nbr))

            if __debug__:
                print(f'unvisited_nodes size:\n{len(unvisited_nodes)}')

            heapq.heappop(unvisited_nodes)
            visited.add(n)

            nonlocal node_counter
            node_counter += 1

        # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - Calling sampling algorithm for each node in current layer')
        while node_counter < self.number_of_nodes:
            layer_counter += 1
            if __debug__:
                print(f'layer_counter: {layer_counter}')

            q1_quota_size = int(self.q1 * len(unvisited_nodes))

            # at least one unvisited node must be visited per iteration of algorithm
            q1_quota_size = max([q1_quota_size, 1])

            # Visiting new nodes
            # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - Adding new nodes to visited and swapping current layer with new layer')
            q1_extract = heapq.nsmallest(n=q1_quota_size, iterable=unvisited_nodes)
            if __debug__:
                print(f'q1_quota_size: {q1_quota_size}')
                print(f'q1_extract: {q1_extract}')
                print(f'unvisited_nodes head (before): {unvisited_nodes[:10]}')

            for q1_new_node in q1_extract:
                _sample_at_node(n=q1_new_node[1])

                if __debug__:
                    print(f'unvisited_nodes head (after): {unvisited_nodes[:10]}')
                    print(f'node_counter: {node_counter}')

            # Brief check for exceeding the node limit for sampling
            if node_counter > self.number_of_nodes:
                break

        visited_list = sorted(list(visited), key=lambda n, graph=graph : graph.degree[n], reverse=True)[:self.number_of_nodes]

        # Forces start_node to be part of visited set
        if start_node not in visited_list:
            visited_list[-1] = start_node

        sampled_network = graph.subgraph(nodes=visited_list)

        if __debug__:
            print(f'visited_list: {visited_list}')
            print(f'sampled_network: {sampled_network}')

        # pdb.set_trace(header='CaterpillarQuotaWalk - sample - _sample_at_node - returning sampled network')
        return sampled_network
#endregion
#endregion

#region
# DEPRECATED
# class RWSampler:
#     """[summary]
#     """

#     def __init__(self, number_of_nodes: int=100):
#         """[summary]

#         Args:
#             number_of_nodes (int, optional): [description]. Defaults to 100.
#         """
#         self.number_of_nodes = number_of_nodes

#         if __debug__:
#             print(f'number_of_nodes: {self.number_of_nodes}')

#     def sample(self, graph: nx.Graph, start_node: int=None):
#         """[summary]

#         Args:
#             graph (nx.Graph): [description]
#             start_node (int, optional): [description]. Defaults to None.
#         """
#         # pdb.set_trace(header='CaterpillarQuotaWalk - sample - Entering sample')
#         if __debug__:
#             print('graph: {graph}')
#             print('start_node: {start_node}')

#         # Node collections
#         visited = set([start_node])
#         curr_layer = set([start_node])
#         next_layer = set()
#         curr_node = start_node

#         # DEBUGGING PURPOSES
#         layer_counter = 0
#         node_counter = 1

#         while node_counter < self.number_of_nodes:
#             unvisited_neighbors = np.asarray(a=[nbr for nbr in graph.neighbors(n=curr_node)])

#             # No more unvisitewd neighbors for current node, so tries a random visited node
#             if unvisited_neighbors.size == 0:
#                 curr_node = random.choice(seq=visited)
#                 continue

#             random_unvisited_neighbor = random.choice(seq=unvisited_neighbors)
#             visited.add(random_unvisited_neighbor)
#             curr_node = random_unvisited_neighbor

#             layer_counter += 1
#             node_counter += 1
#             if __debug__:
#                 print(f'layer_counter: {layer_counter}')
#                 print(f'node_counter: {node_counter}')

#         sampled_network = graph.subgraph(nodes=visited)
#         return sampled_network

        # Choosing nodes to contribute to sampling
        # def _sample_at_node(n: Any):
        #     """[summary]

        #     Args:
        #         n (Any): [description]

        #     Raises:
        #         ValueError: n is not part of the graph's vertex set
        #         ValueError: n is already visited
        #     """
        #     # pdb.set_trace(header='RWSampler - sample - _sample_at_node - Entering _sample_at_node')
        #     if __debug__:
        #         print(f'n: {n}')

        #     if n not in graph:
        #         raise ValueError(f'Node {n} must exist inside network {graph}')
        #     if n not in visited:
        #         raise ValueError(f'Node {n} must be already visited.')

        #     # unvisited neighboring nodes of n rnaked by degree descending


        #     # No unvisited neighbors left
#endregion
