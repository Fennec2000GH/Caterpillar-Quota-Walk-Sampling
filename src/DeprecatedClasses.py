
class NSAgent(Agent):
    """Agent integrated with networkx"""

    def __init__(self, unique_id: int, model: NSModel, node: Any, method: NSMethod) -> None:
        """
        Initializes required attributes under Agent

        Parameters
        :param unique_id: Unique id inherited from mesa.Agent
        :param model: Model inherited from mesa.Model
        :param node: Current node NSAgent is occupying in model
        :param kwargs: Extra properties to assign to NSAgent
        :return: None
        """
        super().__init__(unique_id=unique_id, model=model)

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
            return

        # ATTRIBUTES
        self.__method = method
        self.__node = node
        self.__active = param.Boolean(default=True,
                                    doc='Whether the NSAgent can still respond to step function next time')
        self.__log_visited = param.Boolean(default=False,
                                        doc='Whether to track the visited nodes by this NSAgent in a numpy array')
        self.__visited_nodes = param.Array(default=np.asarray(a=[]), doc='Collects visited nodes when sampling')
        self.__extra_properties = param.Dict(default={}, doc='Extra properties associated with the NSAgent')
        self.visit(next_node=node)

    # PROPERTIES
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
            if new_node not in self.network:
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
    def log_visited(self) -> bool:
        """
        Finds out whether NSAgent keeps record of visited nodes

        :return: True if NSAgent keeps record of visited nodes, else False
        """
        return self.__log_visited

    @log_visited.setter
    def log_visited(self, new_status: bool) -> None:
        """
        Whether to continue tracking visited nodes or not. If set to False, self.__visited_nodes will be cleared.

        :param new_status: True if NSAgent keeps track of visited nodes, else False
        :return: None
        """
        # Checking for valid arguments
        try:
            if type(new_status) != bool:
                raise TypeError('new_status must be of type bool')
        except TypeError as error:
            print(str(error))
            return
        if not new_status:
            self.clear_visited_nodes()
        self.__log_visited = new_status

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

    def get_visited_edges(self) -> Union[np.ndarray, None]:
        """
        Gets numpy array of visited edges

        :return: Numpy array of visited edges traversed by NSAgent during sampling run
        """
        vn = self.__visited_nodes
        if vn.size <= 1:
            return None
        return np.asarray(a=[(vn[i], vn[i + 1]) for i in np.arange(vn.size - 1)], dtype=tuple)

    # MUTATORS
    def visit(self, next_node: Any) -> None:
        """
        Helper function to visit a node by appending that node to self.__visited_nodes

        :param next_node: New node to be visited
        :return: None
        """
        # Checking for valid arguments
        try:
            if next_node not in self.network.nodes:
                raise ValueError('next_node must exist in NSAgent\'s model\'s network')
        except ValueError as error:
            print(str(error))
            return

        self.network.nodes[next_node]['visited'] = True
        if self.__log_visited:
            self.__visited_nodes = np.append(arr=self.__visited_nodes, values=next_node)
        self.__node = next_node

    def set_extra_properties(self, **kwargs) -> None:
        """
        Sets one or more extra properties simultaneously

        Parameters
        :param kwargs: Key-Value pairs to insert as new extra properties
        :return: None
        """
        for key, value in kwargs.items():
            self.__extra_properties.update({str(key): value})

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
        return np.asarray(a=[node for node in self.__network.nodes if self.__network[node]['visited']])

    def get_visited_edges(self) -> np.ndarray:
        """
        Gets numpy array of unique visited edges

        :return: Numpy array of visited edges
        """
        visited_edges = set()
        for agent in self.schedule.agent_buffer(shuffled=False):
            vn = agent.get_visited_nodes()
            for index in np.arange(vn.size - 1):
                visited_edges.add((vn[index], vn[index + 1]))
        return np.asarray(a=list(visited_edges))

    # MUTATORS
    def reset(self) -> None:
        """
        Resets all NSAgents back to start_node with cleared visit history

        :return: None
        """
        for agent in self.schedule.agent_buffer(shuffled=False):
            original_log_visited = bool(copy.deepcopy(x=agent.log_visited))
            agent.log_visited = False  # Clears any visited nodes logged by agent
            agent.log_visited = original_log_visited
            agent.node = self.__start_node

        # Marks all nodes in network as not visited
        for node in self.__network.nodes:
            self.__network.nodes[node]['visited'] = False

    def score(self) -> Optional[float]:
        """
        Scores current sampling of network based on target nodes generated from a target nodes generator from NSMethod
        as an answer sheet

        :return: Score as float
        """
        # Checking for valid arguments
        try:
            if self.__method.score_func is None:
                raise ValueError('score_func in NSMethod cannot be None')
            if not isinstance(self.__method.score_func, Callable):
                raise TypeError('score_func in NSMethod must be of type Callable')
            if not self.__sampled:
                raise ValueError('network must be sampled first before scoring')
        except (TypeError, ValueError) as error:
            print(str(error))
            return

        # Scoring begins
        tng = self.__method.target_nodes_generator
        sf = self.__method.score_func
        sf_params = self.__method.score_func_params
        test_nodes = self.get_visited_nodes()  # np.ndarray of visited nodes for testing
        sf_params['test_nodes'] = test_nodes
        if 'model' in sf_params:
            sf_params['model'] = self
        if 'network' in sf_params:
            sf_params['network'] = self.__network

        # Check for the need of a target node generator
        if tng is not None and 'target_nodes' in sf_params:
            tng_params = self.__method.target_nodes_generator_params
            tng_params['model'] = self
            target_nodes = tng(**tng_params)  # np.ndarray of expected nodes
            sf_params['target_nodes'] = target_nodes

        score = sf(**sf_params)
        return score


    def step(self, n_steps: int, other_func: FuncIterable = None) -> None:
        """
        Activates model to run n steps for each NSAgent. There is an optional feature to call each function in an
        iterable, for which each element is a tuple of function and dictionary of parameters.

        Parameters
        :param n_steps: Number of steps for each NSAgent to step through
        :param other_func: Iterable of or single intermittent function with parameters called after advancing each step
        :return: None
        """
        for _ in np.arange(n_steps):
            self.schedule.step()

            # Executing network-wide function from NSMethod, if applicable
            if self.__method.network_func is not None:
                func = self.__method.network_func
                params = self.__method.network_func_params if self.__method.network_func_params is not None else {}
                if 'model' in params:
                    params['model'] = self
                if 'network' in params:
                    params['network'] = self.__network
                func(**params)

            # Executing potential intermittent function
            if other_func is not None:
                for func_tuple in other_func:
                    func = func_tuple[0]
                    params = func_tuple[1]
                    if 'model' in params:
                        params['model'] = self
                    if 'network' in params:
                        params['network'] = self.__network
                    func(**params)
