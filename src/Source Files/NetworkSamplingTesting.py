
from collections.abc import Iterable
import itertools
import networkx as nx
import numpy as np
from typing import Callable, Dict, Any
from inspect import signature
from NetworkSampling import NSModel, NSMethod

class NSGridSearch:
    """Tool to seek optimal parameter values for network sampling function"""
    def __init__(self, model: NSModel, target_func: Callable, target_func_params: Dict[str, Any], test_func: Callable, test_func_params: Dict[str, Any], score_func: Callable) -> None:
        """
        Initializes gridsearch setup with appropriate functions

        Parameters
        model - NSModel to be scored on accuracy
        target_func - network sampling function used in step for each agent that will generate the expected nodes to be visited
        target_func_params - dict of fixed parameter values for target_func
        test_func - network sampling function used in step for each agent that will generate a set of nodes to be compared
        test_func_params - parameter values for test_func; each value to a key must be an iterable
        against the expected set of nodes
        score_func - scoring function that takes in two arrays of nodes
        """
        try:
            if model.network is None or len(model.network.nodes) == 0:
                raise ValueError("NSModel must have non-None graph with at least one (1) node")
            if target_func is None or test_func is None:
                raise ValueError("Both function inputs must not be None")
        except ValueError as error:
            print(str(error))
            return

        self.__model = model
        self.__target_method = NSMethod(func=target_func, params={'agent': None, **target_func_params})
        self.__test_method = NSMethod(func=test_func, params={'agent': None, **test_func_params})
        self.__score_func = score_func

        # Properties
    @property
    def model(self) -> NSModel:
        """Gets the current NSModel being gridsearched"""
        return self.__model

    @model.setter
    def model(self, new_model: NSModel) -> None:
        """
        Sets a new NSModel to be gridsearched

        Parameters
        new_model - new NSModel
        """
        self.__model = new_model

    def get_target_func(self) -> Callable:
        """Gets the network sampling function that generates expected nodes first"""
        return self.__target_method.func

    def get_test_func(self) -> Callable:
        """Gets the network sampling function that generates nodes to be scored"""
        return self.__test_method.func

    @property
    def score_func(self) -> Callable:
        """Gets the scoring function used to evaluate accuracy"""
        return self.__score_func

    @score_func.setter
    def score_func(self, new_score_func: Callable) -> None:
        """
        Sets new scoring function

        Parameters
        new_score_func - new scoring function to replace current one
        """
        # Error checking for arguments
        sig = signature(new_score_func)
        try:
            if new_score_func is None:
                raise ValueError("new_score_func cannot be None")
            if len(sig.parameters) != 2:
                raise ValueError("new_score_func must take in two (2) parameters only")
        except ValueError as error:
            print(str(error))
            return
        self.__score_func = new_score_func

    # ACCESSORS
    def score(self) -> float:
        """Computes accuracy score for one instance using initialized score function"""
        # Computing visited nodes from target function
        self.__model.reset(method=NSMethod(func=self.__target_func, params={'agent': None, **target_func_params}))
        model.step(n_steps=n_steps)
        vn_target = model.get_visited_nodes()
        vn_target = self.__model.self.__target_func

    # MUTATORS
    def search(n_steps: int, n_jobs: int = 1, ) -> None:
        """
        Executes gridsearch with given function parameters. Each value that is an iterable in test_func_params
        will be iterated over inside nested for loops such that each possible combination of test function
        parameter value will be scored.

        Parameters
        n_jobs - number of trials or jobs to complete; each job is a single run of this search function; if
        n_jobs is more than 1, the mean of scores for each distinct tuple of test_param_values will determine
        the optimal parameter values for test_func

        """
        # Valid type checking
        try:
            if n_jobs < 1:
                raise ValueError("n_jobs must be at least 1")
            if np.all(a=[type(param) != Iterable for param in test_func_params.values()], axis=None):
                raise ValueError("Each parameter for test_func must be an iterable")
        except ValueError as error:
            print(str(error))
            return

        self.__parameters = np.asarray(a=list(test_func_params.keys()), dtype=str)

        # best set of parameter values for each job / trial
        self.__optimal_parameters = np.asarray(a=[], dtype=tuple, shape=n_jobs)

        # Cartesian product i.e. all possible sets of parameter values
        possible_parameters = itertools.product(list(test_func_params.values()))
        best_score = 0 # highest score for a set of parameter values so far
        for job in np.arange(n_jobs):
            for parameters in possible_parameters:


                # Computing visited nodes from test function
                model.reset(method=NSMethod(func=self.__test_func, params={'agent': None,
                                                                           **dict
                                                                               (zip(self.__parameters, list(parameters)))}))
                model.step(n_steps=n_steps)
                vn_test = model.get_visited_nodes()

                # Computing score and updating optimal parameters attribute
                current_score = self.__score_func(vn_target, vn_test)
                if current_score > best_score:
                    best_score = current_score
                    self.__optimal_parameters[job] = (parameters, best_score)
