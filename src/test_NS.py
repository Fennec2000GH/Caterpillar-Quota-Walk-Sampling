
import matplotlib.pyplot as plt
import networkx as nx
import pytest

from NetworkSampling import NSMethod, NSModel
from NetworkSamplingContainer import NSContainer
from NetworkSamplingFunctions import random_walk, forest_fire, snowball, caterpillar_quota_walk

# NSMethod
@pytest.mark.NSMethod
def test_NSMethod() -> None:
    """
    Tests instantiation of NSMethod

    :return: None
    """

    nsmethod_random_walk = NSMethod(func=random_walk, params={})




@pytest.mark




# NSContainer
@pytest.mark.NSContainer
def test_init() -> None:
    """
    Tests the initialization of NSContainer object

    :return: None
    """
    method = NSMethod(func=random_walk, params={'agent': None})
    model = NSModel(method=method,
                    network=nx.complete_graph(n=10, create_using=nx.Graph),
                    n_agents=10,
                    start_node=0)
    nsc = NSContainer(model=model)
    assert True
