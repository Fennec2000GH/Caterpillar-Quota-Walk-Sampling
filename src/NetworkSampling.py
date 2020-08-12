
from __future__ import annotations
from inspect import signature

from mesa import Agent, Model
from mesa.time import BaseScheduler, SimultaneousActivation
import networkx as nx
import numpy as np
import param
from typing import Any, Callable, Dict, NamedTuple


class NSMethod(NamedTuple):
    """
    Model for network sampling method

    Fields
    :param func: Callable or function that must take in at last 'agent' as a parameter, and the return
    value(s) must be 3-element tuple of format (next_node, next_method: NSMethod, pause: Boolean).
    These fields provide the next node in the model's network for this agent to travel to, a
    possible change in network sampling method for next turn, and whether to pause any further 
    activity of this agent, respectively.

    :param params: The parameters and corresponding values to properly execute the provided network
    sampling method. The word 'agent' must be in the keys, and the preset value is preferably
    None. Type must be dict of {str: object} KV pairs.
    """
    func: Callable
    params: Dict[str, Any]

class NSAgent(Agent):
    """Agent integrated with networkx"""

    def __init__(self, unique_id: int, model: NSModel, node: Any, method: NSMethod) -> None:
        """
        Initializes required attributes under Agent

        Parameters
        :param unique_id: Unique id inherited from mesa.Agent
        :param model: Model inherited from mesa.Model
        :param node: Current node NSAgent is occupying in model
        :return: None
        """
        # Checking for valid arguments
        try:
            if type(unique_id) != int:
                raise TypeError('unique_id must be of type int')
            if not isinstance(model, NSModel):
                raise TypeError('model must be of type NSModel')
            if node not in model.network.nodes:
                raise ValueError('node not in model\'s network')
            if not isinstance(method, NSMethod):
                raise TypeError('method must be of type NSMethod')
        except (TypeError, ValueError) as error:
            print(str(error))
            del self

        super().__init__(unique_id=unique_id, model=model)

        self.__active = param.Boolean(bool=True, doc='Whether the NSAgent can still respond to step function next time')
        self.__extra_properties = param.Dict(default=dict(), doc='Extra properties associated with the NSAgent')
        self.__method = method
        self.__node = node
        self.__visited_nodes = param.Array(default=np.asarray(a=[node]), doc='Collects visited nodes when sampling')

    @property
    def active(self) -> bool:
        """
        Indicate whether this NSAgent is active each iteration or paused

        :return: Whether NSAgent has abn active step function
        """
        return self.__active

    @active.setter
    def active(self, state: bool) -> None:
        """
        Set active state

        Parameters
        :param state: Whether this NSAgent is active or not
        :return: None
        """
        self.__active = state

    @property
    def extra_properties(self) -> dict:
        """
        Get entire dict of extra properties

        :return: Dict of extra properties for NSAgent
        """
        return self.__extra_properties

    @extra_properties.setter
    def extra_properties(self, new_extra_properties: Dict[str, Any]) -> None:
        """
        Resets extra properties entirely to another dict

        Parameters:
        :param new_extra_properties: Replacement for current dict of extra properties
        :return: None
        """
        # Checking for valid argument
        try:
            if type(new_extra_properties) != dict:
                raise TypeError('new_extra_properties must be of type dict')
            if not all([type(key) == str for key in new_extra_properties.keys()]):
                raise TypeError('keys in new_extra_properties must be of type str')
        except TypeError as error:
            print(str(error))
            return
        self.__extra_properties = new_extra_properties

    @property
    def method(self) -> NSMethod:
        """
        Get network sampling method employed at each step of ABM

        :return: NSMethod named tuple holding model's current network sampling method
        """
        return self.__method

    @method.setter
    def method(self, new_method: NSMethod) -> None:
        """
        Sets new algorithm / smapling method for agent

        Parameters:
        :param new_method: Replacement for current NSMethod object used as model's network sampling method
        :return: None
        """
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
        """
        Current node or vertex NSAgent owns
        :return: Networkx node NSAgent is located over
        """
        return self.__node

    @node.setter
    def node(self, new_node) -> None:
        """
        Sets new node or vertex for NSAgent to own

        Parameters
        :param new_node: New node for current NSAgent object to be located at
        :return: None
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
    def network(self) -> nx.Graph:
        """
        Gets Networkx object, ie the network to be used in the model

        :return: Networkx graph of model
        """
        return self.model.network

    # ACCESSORS
    def get_extra_property(self, extra_property_name: str, default: Any = None) -> Any:
        """
        Gets value associated with extra property

        Parameters
        :param extra_property_name: Name corresponding to key of new extra property
        :param default: Value to return if extra_property_name does not exist in keys
        :return: Value associated with extra_property_name key, if exists as an extra property
        """
        return self.__extra_properties.get(extra_property_name, default)

    def get_visited_nodes(self) -> np.ndarray:
        """
        Gets numpy array of visited nodes

        :return: Numpy array of visited nodes by NSAgent during sampling run
        """
        return self.__visited_nodes

    def get_visited_edges(self) -> np.ndarray:
        """
        Gets numpy array of visited edges

        :return: Numpy array of visited edges traversed by NSAgent during sampling run
        """
        vn = self.__visited_nodes
        if vn.size <= 1:
            return np.empty(shape=(0,), dtype=tuple)
        return np.asarray(a=[(vn[i], vn[i + 1]) for i in np.arange(vn.size - 1)], dtype=tuple)

    # MUTATORS
    def set_extra_property(self, key: str, value: Any) -> None:
        """
        Sets new extra property KV pair

        Parameters
        :param key: the key
        :param value: the corresponding value
        :return: None
        """
        self.__extra_properties.update({key: value})

    def set_many_extra_properties(self, **kwargs) -> None:
        """
        Sets many extra properties simultaneously

        Parameters
        :param kwargs: Key-Value pairs to insert as new extra properties
        :return: None
        """
        for key, value in kwargs.items():
            self.set_extra_property(key=str(key), value=value)

    def clear_extra_properties(self) -> None:
        """
        Empty out dict of extra properties

        :return: None
        """
        self.__extra_properties.clear()

    def clear_visited_nodes(self) -> None:
        """
        Clears history of visited nodes

        :return: None
        """
        self.__visited_nodes.clear()

    def step(self) -> None:
        """
        What the agent does at each step of ABM

        :return: None
        """
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

    def __init__(self, method: NSMethod, network: nx.Graph, n_agents: int, start_node: Any) -> None:
        """
        Initializes base network

        Parameters
        :param method: NSMethod object to uniformly assign to each NSAgent object initially
        :param network: networkx.Graph object model is based on
        :param n_agents: Number of NSAgent objects to add to schedule
        :param start_node: Node where all NSAgent objects initially reside
        :return: None
        """
        super().__init__()

        # Checking for non-empty graph with nodes
        try:
            if nx.is_empty(G=network):
                raise ValueError('network does not contain any nodes')
            if n_agents < 0:
                raise ValueError('number of agents cannot be negative')
            if start_node not in network.nodes:
                raise ValueError('start node is not in network')
        except ValueError as error:
            print(str(error))
            del self

        # Building up network for model
        self.__method = method
        self.__network = network
        self.__start_node = start_node
        self.schedule = SimultaneousActivation(model=self)
        for ID in np.arange(n_agents):
            a = NSAgent(unique_id=ID, model=self, node=start_node, method=method)
            self.schedule.add(agent=a)

    @property
    def method(self) -> NSMethod:
        """
        Gets network sampling method with parameter values as NSMethod object

        :return: Current NSMethod object assigned to each agent in the model
        """
        return self.__method

    @method.setter
    def method(self, new_method: NSMethod) -> None:
        """
        Sets new networks sampling method and parameter values

        Parameters
        :param new_method: New network sampling method as NSMethod object for replacement
        :return: None
        """
        # Checking for valid NSMethod object
        try:
            if new_method is None:
                raise TypeError('new_method cannot be None')
            if type(new_method) != NSMethod:
                raise TypeError('new_method must be of type NSMethod')
        except TypeError as error:
            print(str(error))
            return
        self.__method = new_method
        for agent in self.schedule.agent_buffer(shuffled=False):
            agent.method = new_method

    @property
    def network(self) -> nx.Graph:
        """
        Base network holding the model

        :return: Networkx graph object that the model currently uses
        """
        return self.__network

    @network.setter
    def network(self, new_network: nx.Graph) -> None:
        """
        Sets new network for model

        Parameters
        :param new_network: new networkx graph for NSAgent objects to traverse through as model
        :return: None
        """
        # Checking for valid NetworkX Graph
        try:
            if type(new_network) != nx.Graph:
                raise TypeError('new_network must be of type networkx.Graph')
        except TypeError as error:
            print(str(error))
            return
        self.__network = new_network

    @property
    def number_of_agents(self) -> int:
        """
        Count of NSAgents used by the model

        :return: None
        """
        return len(self.schedule.agents)

    @property
    def start_node(self) -> Any:
        """
        Gets initialized node that all agents start at

        :return: Node that all NSAgents initially spawn at
        """
        return self.__start_node

    @start_node.setter
    def start_node(self, new_start_node) -> None:
        """
        Resets start node for model

        Parameters:
        :param new_start_node: Replacement for current starting node
        :return: None
        """
        # Checking for valid new start node
        try:
            if new_start_node not in self.network.nodes:
                raise ValueError('new_start_node is not in current network')
        except ValueError as error:
            print(str(error))
            return
        self.__start_node = new_start_node

    # ACESSORS
    def get_visited_nodes(self) -> np.ndarray:
        """
        Gets numpy array of unique visited nodes

        :return: Numpy array of visited nodes
        """
        visited_nodes = set()
        for agent in self.schedule.agent_buffer(shuffled=False):
            for n in agent.get_visited_nodes():
                visited_nodes.add(n)
        return np.asarray(a=list(visited_nodes))

    def get_visited_edges(self) -> np.ndarray:
        """
        Gets numpy array of unique visited edges

        :return: Numpy array of visited edges
        """
        visited_edges = set()
        for agent in self.schedule.agent_buffer(shuffled=False):
            vn = agent.get_visited_nodes()
            for index in np.arange(0, vn.size - 1):
                visited_edges.add((vn[index], vn[index + 1]))
        return np.asarray(a=list(visited_edges))

    # MUTATORS
    def reset(self) -> None:
        """
        Resets all NSAgents back to start_node with cleared visit history

        :return: None
        """
        for agent in self.schedule.agent_buffer(shuffled=False):
            agent.clear_visited_nodes()
            agent.node = self.__start_node

    def step(self, n_steps: int, func: Callable = None, params: Dict[str, Any] = dict()) -> None:
        """
        Activates model to run n steps for each NSAgent

        Parameters
        :param n_steps: Number of steps for each NSAgent to step through
        :param func: Intermittent function called after advancing each step
        :param params: Parameter values to be passed in for func
        :return: None
        """
        for _ in np.arange(n_steps):
            self.schedule.step()

            # Executing potential intermittent function
            if func is not None:
                func(**params)
