
import copy
import networkx as nx
import numpy as np
from NetworkSampling import NSMethod, NSAgent, NSModel

def random_walk(self, agent: NSAgent) -> None:
    """
    Random Walk Algorithm

    Parameters
    agent - NSAgent object that contributes to the sampling
    """
    # Is the sampling method probabilistic or non-probabilistic
    self.probabilistic = True

    # Reconfiguring agents
    nt = agent.get_network()
    neighbors = np.asarray(a=list(nx.neighbors(G=nt, n=agent.node)))
    next_node = np.random.choice(a=neighbors, size=None)
    agent.node = next_node
    agent.visited_nodes = np.append(arr=agent.visited_nodes, values=next_node)

def random_walk_weighted(self, agent: NSAgent) -> None:
    """
    Random Walk Algorithm chooses next vertex based on percentage of vertex weight in neighborhood

    Parameters
    agent - NSAgent object that contributes to the sampling
    """
    # Is the sampling method probabilistic or non-probabilistic
    self.probabilistic = True

    # Reconfiguring agents
    nt = agent.get_network()
    neighbors = np.asarray(a=list(nx.neighbors(G=nt, n=agent.node)))
    weights = np.array([])
    for n in neighbors:
        """Generate list of weights for nodes"""
        if n not in nx.get_node_attributes(G=nt, name='weight'):
            nt.nodes[n]['weight'] = float(0)
        weights = np.append(arr=weights, values=float(nt.nodes[n]['weight']))
    sum_of_weights = np.sum(a=weights)
    weights = weights / sum_of_weights
    next_node = np.random.choice(a=neighbors, p=weights)
    agent.node = next_node
    agent.visited_nodes = np.append(arr=agent.visited_nodes, values=next_node)

# Proposed method
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
    nt = agent.get_network()
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


