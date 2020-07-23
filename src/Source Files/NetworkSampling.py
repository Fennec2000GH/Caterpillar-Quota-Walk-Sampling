
from __future__ import annotations
from inspect import signature
from mesa import Agent, Model
from mesa.time import BaseScheduler, SimultaneousActivation
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict, NamedTuple


class NSMethod(NamedTuple):
    """
    Model for network sampling method

    Fields
    func - callable or function that must take in at last 'agent' as a parameter, and the return
    value(s) must be 3-element tuple of format (next_node, next_method: NSMethod, pause: Boolean).
    These fields provide the next node in the model's network for this agent to travel to, a
    possible change in network sampling method for next turn, and whether to pause any further 
    activity of this agent, respectively.

    params - the parameters and corresponding values to properly execute the provided network
    sampling method. The word 'agent' must be in the keys, and the preset value is preferably
    None. Type must be dict of {str: object} KV pairs.
    """
    func: Callable
    params: Dict[str, Any]


class NXAgent(Agent):
    """Agent integrated with networkx"""

    def __init__(self, unique_id: int, model: NSModel, node, method: NSMethod) -> None:
        """
        Initializes required attributes under Agent

        Parameters
        unique_id - unique id inherited from mesa.Agent
        model - model inherited from mesa.Model
        node - current node NXAgent is occupying in model
        """
        super().__init__(unique_id=unique_id, model=model)
        try:
            if node not in model.network.nodes:
                raise ValueError('node not in model\'s network')
        except ValueError as error:
            print(str(error))
            del self
        self.__active = True
        self.__extra_properties = Dict[str, Any]
        self.__method = method
        self.__node = node
        self.__visited_nodes = np.asarray(a=[node])

    @property
    def active(self) -> bool:
        """Indicate whether this NXAgent is active each iteration or paused"""
        return self.__active

    @active.setter
    def active(self, state: bool) -> None:
        """
        Set active state

        Parameters
        state - boolean indicating whether this NXAgent is active or not
        """
        self.__active = state

    @property
    def extra_properties(self) -> Dict[str, Any]:
        """Get entire dict of extra properties"""
        return self.__extra_properties

    @extra_properties.setter
    def extra_properties(self, new_extra_properties: Dict[str, Any]) -> None:
        """Resets extra properties entirely to another dict"""
        self.__extra_properties = new_extra_properties

    @property
    def method(self) -> NSMethod:
        """Get network sampling method employed at each step of ABM"""
        return self.__method

    @method.setter
    def method(self, new_method: NSMethod) -> None:
        """Sets new algorithm / smapling method for agent"""
        # Error checking for correct function signature
        sig = signature(new_method.func)
        try:
            if len(sig.parameters) == 0:
                raise ValueError('new_method must have at least one parameter')
            if 'agent' not in sig.parameters or 'agent' not in new_method.params:
                raise ValueError('"agent" must be a parameter name')
        except ValueError as error:
            print(error)
            return
        self.__method = new_method

    @property
    def node(self) -> Any:
        """Current node or vertex NXAgent owns"""
        return self.__node

    @node.setter
    def node(self, new_node) -> None:
        """
        Sets new node or vertex for NXAgent to own

        Parameters
        new_node - new node for current NXAgent object to be located at
        """
        # Error checking for valid new node (existing in model) 
        try:
            if new_node not in self.get_network():
                raise ValueError('new node must be present in current model\'s network')
        except ValueError as error:
            print(str(error))
            return
        self.__node = new_node

    @property
    def visited_nodes(self) -> np.ndarray:
        """Array of visited nodes"""
        return self.__visited_nodes

    @visited_nodes.setter
    def visited_nodes(self, new_visited_nodes: np.ndarray) -> None:
        """Sets new history of visited nodes"""
        self.__visited_nodes = new_visited_nodes

    # ACCESSORS
    def get_network(self) -> nx.Graph:
        """Gets Networkx object, ie the network to be used in the model"""
        return self.model.network

    def get_extra_property(self, extra_property_name: str, default: Any = None) -> Any:
        """Gets value associated with extra property"""
        return self.__extra_properties.get(extra_property_name, default)

    # MUTATORS
    def set_extra_property(self, key: str, value: Any) -> None:
        """
        Sets new extra property KV pair

        Parameters
        key - the key
        value - corresponding value
        """
        self.__extra_properties.update({key: value})

    def set_many_extra_properties(self, **kwargs) -> None:
        """
        Sets many extra properties simultaneously

        Parameters
        kwargs - KV pairs to insert as new extra properties
        """
        for key, value in kwargs.items():
            self.set_extra_property(key=str(key), value=value)

    def clear_extra_properties(self) -> None:
        """Empty out dict of extra properties"""
        self.__extra_properties.clear()

    def clear_visited_nodes(self) -> None:
        """Clears history of visited nodes"""
        self.__visited_nodes.clear()

    def step(self) -> None:
        """What the agent does at each step of ABM"""
        # Returns new node(s) and possibly a new algorithm for next time
        # For the second returned value, the algorithm stays the same if True is returned
        # Otherwise if False, the agent stops any more actions and pauses from then on
        if self.__active:
            func = self.__method.func
            params = self.__method.params
            params['agent'] = self
            params['self'] = func
            func(**params)


# NS = Network Sampling
class NSModel(Model):
    """Model integrated with networkx and base class for random walks"""

    def __init__(self, method: NSMethod, network: nx.Graph, num_agents: int, start_node) -> None:
        """
        Initializes base network

        Parameters
        method - NSMethod object to uniformly assign to each NXAgent object initially
        network - nx.Graph object model is based on
        num_agents - number of NXAgent objects to add to schedule
        start_node - node where all NXAgent objects initially reside
        """
        super().__init__()

        # Checking for non-empty graph with nodes
        try:
            if nx.is_empty(G=network):
                raise ValueError('network does not contain any nodes')
            if num_agents < 0:
                raise ValueError('number of agents cannot be negative')
            if start_node not in network.nodes:
                raise ValueError('start node is not in network')
        except ValueError as error:
            print(str(error))
            del self

        # Building up network for model
        self.__network = network
        self.__start_node = start_node
        self.schedule = SimultaneousActivation(model=self)
        for ID in np.arange(num_agents):
            a = NXAgent(unique_id=ID, model=self, node=start_node, method=method)
            self.schedule.add(agent=a)

    @property
    def network(self) -> nx.Graph:
        """Base network holding the model"""
        return self.__network

    @network.setter
    def network(self, new_network: nx.Graph) -> None:
        """
        Sets new network for model

        Parameters
        new_network - new networkx graph for NXAgent objects to traverse through as model
        """
        self.__network = new_network

    @property
    def number_of_agents(self) -> int:
        """Count of NXAgents used by the model"""
        return self.agents.size

    @property
    def start_node(self):
        """Gets initialized node that all agents start at"""
        return self.__start_node

    @start_node.setter
    def start_node(self, new_start_node) -> None:
        """Resets start node for model"""
        try:
            if new_start_node not in self.network.nodes:
                raise ValueError('new_start_node is not in current network')
        except ValueError as error:
            print(str(error))
            return
        self.__start_node = new_start_node

    # ACESSORS
    def get_visited_nodes(self) -> np.ndarray:
        """Gets array of unique visited nodes"""
        visited_nodes = set()
        for agent in self.schedule.agent_buffer(shuffled=False):
            for n in agent.visited_nodes:
                visited_nodes.add(n)
        return np.asarray(a=list(visited_nodes))

    def get_visited_edges(self) -> np.ndarray:
        """Gets array of unique visited edges"""
        visited_edges = set()
        for agent in self.schedule.agent_buffer(shuffled=False):
            vn = agent.visited_nodes
            for index in np.arange(0, vn.size - 1):
                visited_edges.add((vn[index], vn[index + 1]))
        return np.asarray(a=list(visited_edges))

    # MUTATORS
    def reset(self, method: NSMethod = None) -> None:
        """
        Resets all NXAgents back to start_node with cleared visit history

        Parameters
        method - potentially new networks sampling method to use; otherwise, None indicates no change
        """
        for agent in self.schedule.agent_buffer(shuffled=False):
            agent.clear_visited_nodes()
            agent.node = self.__start_node
            if method is not None:
                agent.method = method

    def step(self, n_steps: int, func: Callable = None, params: Dict[str, Any] = None) -> None:
        """
        Activates model to run n steps for each NXAgent

        Parameters
        n_steps - number of steps for each NXAgent to step through
        model_func - intermittent function called after advancing each step
        params -
        """
        for step_number in np.arange(n_steps):
            self.schedule.step()

            # Executing potential intermittent function
            if func is not None:
                func(**params)
