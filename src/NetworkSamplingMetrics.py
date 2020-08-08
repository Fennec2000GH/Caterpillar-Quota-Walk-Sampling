
# DEPRECATED CODE

from collections.abc import Iterable
import itertools
import numpy as np
from typing import Callable, Dict, Any
from inspect import signature
from NetworkSampling import NSModel, NSMethod

class NSGridSearch:
    """Tool to seek optimal parameter values for network sampling function"""
    def __init__(self, model: NSModel, target_func: Callable, target_func_params: Dict[str, Any], test_func: Callable, test_func_params: Dict[str, Any], test_params: Dict[str, Any]) -> None:
        """
        Initializes gridsearch setup with appropriate functions

        Parameters
        model - NSModel to be scored on accuracy
        target_func - network sampling function used in step for each agent that will generate the expected nodes to be visited
        target_func_params - dict of fixed parameter values for target_func; do not include 'agent' as a parameter
        test_func - network sampling function used in step for each agent that will generate a set of nodes to be compared
        test_func_params - parameter values for test_func setup; do not include 'agent' as a parameter
        test_params - 2D arr
        """
        try:
            if model.network is None or len(model.network.nodes) == 0:
                raise ValueError('NSModel must have non-None graph with at least one (1) node')
            if target_func is None or test_func is None:
                raise ValueError('Both function inputs must not be None')
            if np.any(a=[type(param) != Iterable for param in test_params.values()], axis=None):
                raise ValueError('Each parameter except "agent" in test_params must be an iterable')
        except ValueError as error:
            print(str(error))
            return

        self.__model = model
        self.__target_func = target_func
        self.__test_func = test_func
        self.__setup_params = np.asarray(a=[dict() if target_func_params is None else target_func_params,
                                            dict() if test_func_params is None else test_func_params])
        self.__test_params = test_params

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

    # ACCESSORS
    def get_target_func(self) -> Callable:
        """Gets the network sampling function that generates expected nodes first"""
        return self.__target_func

    def get_test_func(self) -> Callable:
        """Gets the network sampling function that generates nodes to be scored"""
        return self.__test_func

    # MUTATORS
    def set_target_func(self, new_target_func: Callable, new_target_func_params: dict) -> None:
        """
        Sets new function to generate target visited nodes

        Parameters
        new_target_func - new function to replace existing target function
        new_target_func_params - new dict of parameter values for new target function; do not include 'agent' as a parameter
        """
        # Error checking for argument
        try:
            if new_target_func is None:
                raise ValueError('new_target_func cannot be None')
            if np.any(a=[type(param) != Iterable for param in new_target_func_params.values()], axis=None):
                raise ValueError('Each parameter except "agent" for test_func must be an iterable')
        except ValueError as error:
            print(str(error))
        self.__target_func = new_target_func
        self.__setup_params[0] = {'agent': None, **new_target_func_params}

    def set_test_func(self, new_test_func: Callable, new_test_func_params: dict) -> None:
        """
        Sets new function to generate visited nodes to be tested for accuracy

        Parameters
        new_test_func - new function to replace existing test function
        new_test_func_params - new dict of parameter values for new test function, which must be only of type Iterable;
        do not include 'agent' as a parameter
        """
        # Error checking for argument
        try:
            if new_test_func is None:
                raise ValueError('new_test_func cannot be None')
        except ValueError as error:
            print(str(error))
        self.__test_func = new_test_func
        self.__setup_params[1] = {'agent': None, **new_test_func_params}

    def search(self, score_func: Callable, n_steps: int, n_jobs: int = 1) -> dict:
        """
        Executes gridsearch with given function parameters. Each value that is an iterable in test_func_params
        will be iterated over inside nested for loops such that each possible combination of test function
        parameter value will be scored.

        Parameters
        score_func - scoring function that takes in only two parameters - iterable of nodes to be scored and iterable of
        nodes that are expected / correct
        n_steps - number of steps taken by each agent per scoring trial / job of the model
        n_jobs - number of trials or jobs to complete; each job is a single run of this search function; if
        n_jobs is more than 1, the mean of scores for each distinct tuple of test_param_values will determine
        the optimal parameter values for test_func
        """
        # Valid type checking
        sig = signature(obj=score_func)
        try:
            if len(sig.parameters) != 2:
                raise ValueError('score_func must have exactly 2 parameters')
            if n_steps < 1:
                raise ValueError('n_steps must be at least 1')
            if n_jobs < 1:
                raise ValueError('n_jobs must be at least 1')
        except ValueError as error:
            print(str(error))
            return

        # names of parameters to be used in grid search
        parameter_names = np.asarray(a=[param for param in self.__test_params.keys()])

        # Cartesian product i.e. all possible sets of parameter values
        possible_parameters = np.asarray(a=itertools.product(list(self.__test_params.values())))

        # score tracking for each job / trial
        scores = np.array(shape=(possible_parameters.size, n_jobs), dtype=float)

        for job in np.arange(n_jobs):
            indexes = np.arange(possible_parameters.size) # set 0-based index over Cartesian product of all test parameter values
            np.random.shuffle(x=indexes)
            for index in indexes:
                parameter_set  = possible_parameters[index]

                # Computing visited nodes from target function
                self.__model.reset(method=NSMethod(func=self.__target_func, params={'agent': None, **self.__setup_params[0]}))
                self.__model.step(n_steps=n_steps)
                vn_target = self.__model.get_visited_nodes()

                # Computing visited nodes from test function
                self.__model.reset(method=NSMethod(func=self.__test_func, params={'agent':None, **self.__setup_params[1], **dict(zip(parameter_names, parameter_set))}))
                self.__model.step(n_steps=n_steps)
                vn_test = self.__model.get_visited_nodes()

                scores[index, job] = score_func(vn_target, vn_test)

                # Computing mean of scores for each set of parameters over all its trials / jobs
                mean_scores = np.mean(a=scores, axis=1)
                max_mean_index = np.argmax(a=mean_scores)
                return dict(zip(parameter_names, possible_parameters[max_mean_index]))