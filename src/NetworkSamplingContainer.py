
from multimethod import multimethod, isa
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict

from NetworkSampling import NSMethod, NSModel

class NSContainer:
    """Wraps NSAgents, NSModel, and scoring function into a single container"""

    def __init__(self, model: NSModel) -> None:
        """
        Isolated container to run NSModel with the ease of changing NSMethod (network sampling method) and/or score
        function before an individual run without having to explicitly employ setters over and over again.

        :param model: NSModel to run
        :return: None
        """
        # Checking for valid model
        try:
            if model is None:
                raise TypeError('model cannot be None')
            if type(model) != NSModel:
                raise TypeError('model must of type NSModel')
        except TypeError as error:
            print(str(error))
            return
        self.__model = model
        self.__method = self.__model.method

    # PROPERTIES
    @property
    def model(self) -> NSModel:
        """Gets current model"""
        return self.__model

    @model.setter
    def model(self, new_model: NSModel) -> None:
        """Sets new model"""
        # Checking for valid model
        try:
            if new_model is None:
                raise TypeError('model cannot be None')
            if type(new_model) != NSModel:
                raise TypeError('model must of type NSModel')
        except TypeError as error:
            print(str(error))
            return
        self.__model = new_model

    # METHODS
    @multimethod
    def run(self, n_steps: isa(int)) -> None:
        """
        Runs current model

        Parameters
        :param n_steps:
        :return: None
        """
        self.__model.reset()
        self.__model.step(n_steps=n_steps)

    @multimethod
    def run(self, n_steps: isa(int), func: isa(Callable), params: isa(Dict[str, Any])) -> None:
        """
        Runs current model but temporarily switches NSMethod to another given network sampling method

        :param n_steps: Number of steps for each NSAgent to step through
        :param func: Network sampling function
        :param params: Dict of parameter values for func; must have 'agent' as a parameter
        :return: None
        """
        # Checking for valid func and params
        try:
            if func is None:
                raise TypeError('funccannot be None')
            if 'agent' not in params:
                raise ValueError('"agent" must be a key in params')
        except (TypeError, ValueError) as error:
            print(str(error))
            return
        self.__model.reset()
        self.__model.method = NSMethod(func=func, params=params)
        self.__model.step(n_steps=n_steps)

    @multimethod
    def score(self, n_steps: isa(int), score_func: isa(Callable), score_params: isa(Dict[str, Any]), target_nodes: isa(np.ndarray)) -> float:
        """
        Runs and scores current model after running each NSAgent n_steps

        Parameters
        :param n_steps: Number of steps for each NSAgent to step through
        :param score_func: Scoring function to be used; must take in at least 2 np.ndarray parameters for target nodes
        and test nodes
        :param score_params: Dict of any other fixed parameter values for scoreing function to be complete
        :param target_nodes: Numpy array of expected nodes to be visited and evaluated against
        :return: Score from evaluating accuracy of current NSModel using provided scoring function
        """
        self.__model.reset()
        self.__model.step(n_steps=n_steps)
        test_nodes = self.__model.get_visited_nodes()
        return score_func(**{'target_nodes': target_nodes, 'test_nodes': test_nodes, **score_params})

    @multimethod
    def score(self, n_steps: isa(int), score_func: isa(Callable), score_params: isa(Dict[str, Any]), target_nodes: isa(np.ndarray),
              func: isa(Callable), params: isa(Dict[str, Any])) -> float:
        """
        Runs and scores current model after running each NSAgent n_steps, temporarily using a different network sampling
        function than provided during initialization of NSContainer

        Parameters
        :param n_steps: Number of steps for each NSAgent to step through
        :param score_func: Scoring function to be used; must take in at least 2 np.ndarray parameters for target nodes
        and test nodes
        :param score_params: Dict of any other fixed parameter values for scoreing function to be complete
        :param target_nodes: Numpy array of expected nodes to be visited and evaluated against
        :param func: Temporary new network sampling function to be used when scoring this run of the current NSModel
        :param params: Dict of parameter values for func; must contain at least 'agent' as a key
        :return: Score from evaluating accuracy of current NSModel using provided scoring function
        """
        # Checking for valid func and params
        try:
            if func is None:
                raise TypeError('func cannot be None')
            if 'agent' not in params:
                raise ValueError('"agent" must be a key in params')
        except (TypeError, ValueError) as error:
            print(str(error))
            return
        self.__model.reset()
        self.__model.method = NSMethod(func=func, params=params)
        self.__model.step(n_steps=n_steps)
        test_nodes = self.__model.get_visited_nodes()
        return score_func(**{'target_nodes': target_nodes, 'test_nodes': test_nodes, **score_params})

