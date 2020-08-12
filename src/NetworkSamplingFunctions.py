
from collections import Iterable
import copy
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict, Union

from NetworkSampling import NSMethod, NSAgent

# Random Walk Algorithm with adjustable parameters
def random_walk(self, agent: NSAgent, p: Callable = None, p_params: Dict[str, Any] = None, lazy: float = 0.0) -> None:
    """
    Random Walk Algorithm function is modifiable. The algorithm chooses next vertex based on a probability distribution
    of weights corresponding to the neighborhood. This distribution can be simply the percentage makeup of each adjacent
    node to the sum of all adjacent nodes in the neighborhood, but any function that returns an iterable of numbers will
    work. The function for p must take in a list of nodes from calling NetworkX.neighbors as first parameter. If p is
    None, then the probability distribution is uniform for all adjacent nodes, and the algorithm simply becomes the
    classic random walk.

    Parameters
    :param self: self
    :param agent: NSAgent object that contributes to the sampling
    :param p: Function returning iterable representing distribution of weights for adjacent nodes
    :param p_params: Any other parameters needed to make function p complete
    :lazy: Probability of NSAgent staying in current node for this step without visiting any adjacent nodes
    :return:
    """
    # Checking for valid arguments
    try:
        if p is not None and isinstance(p, Callable):
            raise TypeError('p must either be None or a Callable')
        if isinstance(p, Callable) and type(p_params) != dict:
            raise TypeError('p_params must be of type dict')
        if type(p_params) == dict and not all([type(key) == str for key in p_params.keys()]):
            raise TypeError('all keys in p_param must be of type str')
    except TypeError as error:
        print(str(error))
        return

    # Is the sampling method probabilistic or non-probabilistic
    self.probabilistic = True

    # Is the random walk lazy
    self.lazy = True if lazy > 0 else False

    # Reconfiguring agents
    nt = agent.network
    neighbors = np.asarray(a=list(nx.neighbors(G=nt, n=agent.node)))
    p_params = {} if p_params is None else p_params
    weights = p(neighbors, **p_params) if p is not None else np.repeat(a=1 / neighbors.size, repeats=neighbors.size)

    # Checking for iterable type for weights
    try:
        if not isinstance(weights, Iterable):
            raise TypeError('return type of Callable p must be of type Iterable')
    except TypeError as error:
        print(str(error))
        return

    next_node = np.random.choice(a=neighbors, p=weights)
    if float(np.random.random(size=None)) > lazy:
        agent.node = next_node
        agent.visited_nodes = np.append(arr=agent.visited_nodes, values=next_node)

def forest_fire(self, agent: NSAgent, p_visit: Union[float, Callable], p_params: Dict[str, Any] = None, p_self_ignite = 0.1) -> None:
    """
    This version of Leskovec's forest fire sampling algorithm is adapted for networks / graphs. Each currently visited
    node has probability p_visit for its agent to visit an adjacent node. If p_visit is a function, it must take in
    an iterable of the adjacent nodes (neighbors) created from NetworkX.neighbors as the first parameter and return an
    iterable of numbers as weights corresponding to the iterable of neighbors.

    :param self: self
    :param agent: NSAgent object that contributes to the sampling
    :param p_visit: Probability that an adjacent unvisited node becomes visited
    :param p_params:
    :param p_self_ignite: Probability that an unvisited node spontaneously self visits
    :return: None
    """
    # Is the sampling method probabilistic or non-probabilistic
    self.probabilistic = True


    nt = agent.network
    neighbors = np.asarray(a=list(nx.neighbors(G=nt, n=agent.node)))

    # Another function to run in the background each time NSModel calls step
    # This is to execute spontaneous self-ignition based on p_self_ignite
    def self_ignite() -> None:
        """Executes probabilistic self ignition of unvisited nodes anywhere in network"""
        for nbr in neighbors:
            # Whether for nbr to self-ignite or not
            if np.random.random(size=None) < p_self_ignite:
                


    self.background_func =

    p_params = {} if p_params is None else p_params
    weights = p_visit(neighbors, **p_params) if p_params is not None else np.repeat(a=1 / neighbors.size, repeats=neighbors.size)

    # Checking for valid weights
    try:
        if not isinstance(weights, Iterable):
            raise TypeError('return type of Callable p must be of type Iterable')
        if np.sum(a=weights) != 1:
            raise ValueError('weights must sum to 1')
    except (TypeError, ValueError) as error:
        print(str(error))
        return

    nsmethod = NSMethod(func=forest_fire, params={'agent': None, 'p_visit': p_visit, 'p_params': p_params, 'p_self_ignite': p_self_ignite})
    for index, nbr in enumerate(neighbors, start=0):
        # Whether to visit neighbor at this index given its probability of visit
        if np.random.random(size=None) < weights[index]:
            schedule = agent.model.schedule
            new_agent = NSAgent(unique_id=schedule.get_agent_count() + 1, model=agent.model, node=nbr, method=nsmethod)
            new_agent.visited_nodes = np.append(arr=agent.get_visited_nodes(), values=nbr)
            schedule.add(a=new_agent)
            agent.active = False

# def snowball(self, agent: NSAgent, n: int) -> None:
#     """
#
#     :param self: self
#     :param agent: NSAgent object that contributes to the sampling
#     :param n: Fixed number of adjacent unvisited nodes of NSAgent to visit next, or all neighbors if degree of NSAgent
#     is less than or equal to n
#     :return: None
#     """
#
#     neighbors_remaining = copy.deepcopy(x=neighbors)
#
#     for _ in np.arange(neighbors.size):
#         next_node_index = np.random.choice(a=np.arange(neighbors_remaining.size), p=weights)
#         np.neighbors_remaining = np.delete(arr=neighbors_remaining, obj=next_node_index, axis=None)
#         next_node = neighbors[next_node_index]
#         next_node
#         agent.visited_nodes = np.append(arr=agent.visited_nodes, values=next_node)

# PROPOSED METHOD
def caterpillar_quota_walk(self, agent: NSAgent, Q1: float, Q2: float) -> None:
    """
    Proposed algorithm takes in two percentage quotas (Q1, Q2 such that Q1 < Q2 <= 100%). The
    minimal quorum of weighted neighbors, when ranked from highest to lowest weight, that
    cumulatively meets or exceeds Q1 continue to extend the central path(s) of the sampled
    subgraph. Analogously, the minimal quorum of weighted neighbors that cumulatively meet or
    exceed Q2, but minus the neighbors already included in Q1, extend as single-edge branches
    of the sampled subgraph based on the caterpillar tree graph model.

    Parameters
    agent - NSAgent object that contributes to the sampling
    """
    # Is the sampling method probabilistic or non-probabilistic
    self.probabilistic = False

    # Reconfiguring agents
    nt = agent.network
    neighbors = list(nx.neighbors(G=nt, n=agent.node))
    for n in neighbors:
        """Patch any neighboring node without 'weight' attribute"""
        if n not in nx.get_node_attributes(G=nt, name='weight'):
            nt.nodes[n]['weight'] = float(0)

    # Neighboring nodes ranked from highest to lowest weights
    neighbors_high_to_low = sorted(neighbors, key=lambda neighbor: nt.nodes[neighbor]['weight'], reverse=True)
    weights = [nt.nodes[n]['weight'] for n in neighbors_high_to_low] # Ranked from highest to lowest
    sum_weights = np.sum(a=weights, axis=None, dtype=float) # Sum of all weights from neighboring nodes

    # Choosing nodes to contribute to sampling
    Q1_quota_weight = Q1 * sum_weights # Expected cumulative weight of Q1 quota
    Q2_quota_weight = Q2 * sum_weights # Expected cumulative weights of Q2 quota
    Q1_index, Q2_index = None, None
    partial_sum = float(0)
    Q1_index_captured = False
    for index, weight in enumerate(weights, start=0):
        partial_sum += weight
        if partial_sum >= Q1_quota_weight and not Q1_index_captured:
            Q1_index = index
            Q1_index_captured = True
        if partial_sum >= Q2_quota_weight:
            Q2_index = index
            break

    # Adding new nodes to network sample
    schedule = agent.model.schedule
    nsmethod = NSMethod(func=caterpillar_quota_walk, params={'agent': None, 'Q1': Q1, 'Q2': Q2})
    for index in np.arange(0, Q1_index + 1):
        next_node = neighbors_high_to_low[index]
        nt.nodes[next_node]['Central Axis'] = True
        new_agent = NSAgent(unique_id=schedule.get_agent_count() + 1, model=agent.model, node=next_node, method=nsmethod)
        new_agent.extra_properties = copy.deepcopy(x=agent.extra_properties, memo=None)
        new_agent.visited_nodes = np.append(arr=agent.visited_nodes, values=next_node)
        schedule.add(agent=new_agent)

    for index in np.arange(Q1_index + 1, Q2_index + 1):
        next_node = neighbors_high_to_low[index]
        nt.nodes[next_node]['Central Axis'] = False
        new_agent = NSAgent(unique_id=schedule.get_agent_count() + 1, model=agent.model, node=next_node, method=nsmethod)
        new_agent.extra_properties = copy.deepcopy(x=agent.extra_properties, memo=None)
        new_agent.visited_nodes = np.append(arr=agent.visited_nodes, values=next_node)
        new_agent.active = False
        schedule.add(agent=new_agent)

    # De-activating current NSAgent object
    agent.active = False


