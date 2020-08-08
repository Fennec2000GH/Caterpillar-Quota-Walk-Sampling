
import networkx as nx
import numpy as np
from typing import Any, Callable, Dict

from NetworkSampling import NSModel

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
    def run(self, ) -> np.ndarray:
        """
        Runs current model and gets numpy array of visited nodes

        :return: Numpy array of visited nodes from one run of model
        """
        self.__model.run_model()

    def run_score(self) -> float:
