
# # DEPRECATED
# import copy
# from mesa.time import SimultaneousActivation
# from multimethod import multimethod
# import networkx as nx
# import numpy as np
# import param
# from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

# from NetworkSampling import NSAgent, NSMethod, NSModel


# class NSContainer:
#     """Wraps NSAgents, NSModel, and scoring function into a single container"""
#     # CLASS ATTRIBUTES
#     __containers = param.Dict(default=dict(), allow_None=False, doc='Keeps track of all NSContainer instances')

#     def __init__(self, unique_name: str, model: NSModel) -> None:
#         """
#         Isolated container to run NSModel with the ease of changing NSMethod (network sampling method) and/or score
#         function before an individual run without having to explicitly employ setters over and over again.

#         :param unique_name: Unique name used as key when storing instance of NSContainer
#         :param model: NSModel to run
#         :return: None
#         """
#         # Checking for valid model
#         try:
#             if model is None:
#                 raise TypeError('model cannot be None')
#             if type(model) != NSModel:
#                 raise TypeError('model must of type NSModel')
#         except TypeError as error:
#             print(str(error))
#             return
#         self.__model = model

#     # PROPERTIES
#     @property
#     def model(self) -> NSModel:
#         """Gets current model"""
#         return self.__model

#     @model.setter
#     def model(self, new_model: NSModel) -> None:
#         """Sets new model"""
#         # Checking for valid model
#         try:
#             if new_model is None:
#                 raise TypeError('model cannot be None')
#             if type(new_model) != NSModel:
#                 raise TypeError('model must of type NSModel')
#         except TypeError as error:
#             print(str(error))
#             return
#         self.__model = new_model

#     # ACCESSORS
#     @classmethod
#     def get_instances(cls) -> np.ndarray:
#         """
#         Gets numpy array of currently existing NSContainer instances

#         :return: Numpy array of NSContainer objects, or None if empty
#         """
#         return cls.__containers if len(cls.__containers) > 0 else None

#     # MUTATORS
#     @classmethod
#     def remove_instances(cls, instances: Union[Iterable[str], str]) -> None:
#         """
#         Remove each NSContainer instance in iterable if an iterable of names is given, or removes a single instance by
#         name if str is given.

#         :param instances: Iterable of names or single name as str of NSContainer instance
#         :return: None
#         """
#         try:
#             if isinstance(instances, Iterable) and not all([type(unique_name) == str for unique_name in instances]):
#                 raise TypeError('Each element in instances must be of type str')
#             if isinstance(instances, Iterable):
#                 keys_not_found = set(instances).difference(set(cls.__containers.keys()))
#                 if len(keys_not_found) > 0:
#                     raise ValueError(f'{keys_not_found} are keys that do not associate with any existing NSContainers')
#         except (TypeError, ValueError) as error:
#             print(str(error))
#         for unique_name in instances:
#             del cls.__containers[unique_name]

#     @classmethod
#     def clear_instances(cls) -> None:
#         """
#         Clears out all instances of NSContainers

#         :return: None
#         """
#         cls.__containers.clear()

#     @multimethod
#     def run(self, n_steps: int) -> None:
#         """
#         Runs current model

#         Parameters
#         :param n_steps:
#         :return: None
#         """
#         self.__model.reset()
#         self.__model.step(n_steps=n_steps)

#     @multimethod
#     def run(self, n_steps: int, return_model: bool = False, **kwargs) -> Optional[NSModel]:
#         """
#         Runs current model uses another NSMethod, network, number of agents, or start node. The model will be a deep
#         copy of the original NSModel used during initialization of NSContainer instance

#         Parameters
#         :param n_steps: Number of steps for each NSAgent to step through
#         :param kwargs: Optional NSModel arguments to temporarily replace certain properties during run
#         :param return_model: Whether to return the temporarily made NSModel with modified properties that is run
#         :return: None
#         """
#         # Checking for valid func and params
#         try:
#             if type(n_steps) != int:
#                 raise TypeError('n_steps must be of type int')
#             if type(return_model) != bool:
#                 raise TypeError('return_model must be of type bool')
#         except TypeError as error:
#             print(str(error))
#             return

#         model_temp = copy.deepcopy(x=self.__model)
#         model_temp.method = kwargs.get(k='method', default=self.__model.method)
#         model_temp.network = kwargs.get(k='network', default=self.__model.network)
#         model_temp.start_node = kwargs.get(k='start_node', default=self.__model.start_node)
#         if 'n_agents' in kwargs:
#             model_temp.schedule = SimultaneousActivation(model=model_temp)
#             for ID in np.arange(kwargs['n_agents']):
#                 a = NSAgent(unique_id=int(ID), model=model_temp, node=model_temp.start_node, method=model_temp.method)
#                 model_temp.schedule.add(agent=a)
#         model_temp.step(n_steps=n_steps)
#         if return_model:
#             return model_temp

#     @multimethod
#     def run_score(self, n_steps: int) -> float:
#         """
#         Runs and scores current model after running each NSAgent n_steps, using the scoring function and target nodes
#         generator from NSMethod

#         Parameters
#         :param n_steps: Number of steps for each NSAgent to step through
#         :return: Score from evaluating accuracy of current NSModel using provided scoring function
#         """
#         self.__model.reset()
#         self.__model.step(n_steps=n_steps)
#         return self.__model.score()

#     @multimethod
#     def run_score(self, n_steps: int, return_model: bool = False, **kwargs) -> Optional[Tuple[float, NSModel]]:
#         """
#         Runs and scores current model after running each NSAgent n_steps, temporarily using different properties than
#         the NSModel created during initialization of NSContainer. Similar to run function that accepts kwargs, a
#         different NSModel, network, number of agents, and / or start node can be used. A deep copy of the original
#         NSModel is made to preserve original properties.

#         Parameters
#         :param n_steps: Number of steps for each NSAgent to step through
#         :param return_model: Whether to return the temporarily made NSModel with modified properties that is run
#         :return: Score from evaluating accuracy of current NSModel using provided scoring function
#         """
#         # Checking for valid arguments
#         try:
#             if type(return_model) != bool:
#                 raise TypeError('return_model must be of type bool')
#         except TypeError as error:
#             print(str(error))
#             return

#         # Running and scoring NSModel's sample
#         model_temp = self.run(n_steps=n_steps, return_model=True, kwargs=kwargs)
#         return model_temp.score(), model_temp if return_model else None

