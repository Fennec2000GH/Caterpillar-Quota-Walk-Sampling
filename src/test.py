
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
import timeit
from typing import Callable

from NetworkSampling import NSAgent, NSModel, NSMethod
from NetworkSamplingFunctions import random_walk, random_walk_weighted, caterpillar_quota_walk
from NetworkSamplingMetrics import NSGridSearch
from NetworkSamplingScoring import zero_one_loss

# BENCHMARK TIME TESTS
@pytest.mark.time
def test_NSMethod_random_walk() -> float:
    """
    Tests the time it takes to setup an NSMethod object based on random walk sampling method, 1000 times

    :return: cumulative time it takes to set up NSMethod for random walk
    """
    stmt = """NSMethod(func=random_walk, params={'agent': None})"""
    time = timeit.timeit(stmt=stmt, setup='', number=1000, globals=globals())
    return time

@pytest.mark.time
def test_NSMethod_random_walk_weighted() -> float:
    """
    Tests the time it takes to setup an NSMethod object based on weighted random walk sampling method, 1000 times


    :return: cumulative time it takes to set up NSMethod for weighted random walk
    """
    stmt = """NSMethod(func=random_walk_weighted, params={'agent': None})"""
    time = timeit.timeit(stmt=stmt, setup='', number=1000, globals=globals())
    return time

@pytest.mark.time
def test_NSMethod_caterpillar_quota_walk() -> float:
    """
    Tests the time it takes to setup an NSMethod object based on caterpillar quota sampling method, 1000 times

    :return: cumulative time it takes to set up NSMethod for caterpillar quota
    """
    stmt = """NSMethod(func=caterpillar_quota_walk, params={'agent': None, 'Q1': 0.8, 'Q2': 0.5})"""
    time = timeit.timeit(stmt=stmt, setup='', number=1000, globals=globals())
    return time

def plot_NSMethod_time() -> None:
    """
    Plots timeit benchmark performance of setting up NSMethod objects on bar plot

    :return: None
    """
    x = np.asarray(a=['Random Walk', 'Weighted Random Walk', 'Caterpillar Quota'], dtype=str)
    height = np.asarray(a=[test_NSMethod_random_walk(),
                           test_NSMethod_random_walk_weighted(),
                           test_NSMethod_caterpillar_quota_walk()],
                        dtype=float)
    plt.figure(figsize=(20, 20), edgecolor='black', facecolor='gray')
    plt.bar(x=x, height=height, color='green')
    plt.title(label='Timeit Benchmark on NSMethod Setup', size=24)
    plt.xlabel(xlabel='Network Sampling Method', size=12)
    plt.ylabel(ylabel='Cumulative Time over 1000 Trials (ms)', size=12)
    plt.grid(b=True, which='major', axis='both')
    plt.show()

@pytest.mark.time
@pytest.mark.parametrize(argnames=['func', 'params'],
                         argvalues=[(random_walk, {'agent': None}),
                                    (random_walk_weighted, {'agent': None}),
                                    (caterpillar_quota_walk, {'agent': None, 'Q1': 0.8, 'Q2': 0.5})])
def test_NSModel_step(func: Callable, params: dict) -> np.ndarray:
    """
    Computes mean time needed for network sampling method to execute in 5, 10, 25, and 50 steps, each using 1000 trials

    :return: Numpy array of mean time for running 5, 10, 25, and 50 steps for a specific network sampling method
    """
    method = NSMethod(func=func, params=params)
    network = nx.complete_graph(n=10, create_using=nx.Graph)

    step_mean_time = np.zeros(shape=4, dtype=float)
    for index, n_steps in enumerate(np.asarray(a=[5, 10, 25, 50]), start=0):
        stmt = """model.step(n_steps=n_steps); model.reset()"""
        setup = """model = NSModel(method=method, network=network, n_agents=10, start_node=0)"""
        time = timeit.timeit(stmt=stmt, setup=setup, number=1000, globals=globals())
        mean_time = time / 1000
        step_mean_time[index] = mean_time
    return step_mean_time

def plot_NSModel_step_time() -> None:
    """
    Plots the mean times for executing 5, 10, 25, and 50 steps for each network sampling method


    :return: None
    """
    plt.figure(figsize=(40, 40), edgecolor='black', facecolor='gray')
    x = np.asarray(a=[5, 10, 25, 50])

    y = test_NSModel_step(func=random_walk, params={'agent': None})
    plt.plot(x=x, y=y, color='violet', marker='square', markersize=12)

    y = test_NSModel_step(func=random_walk_weighted, params={'agent': None})
    plt.plot(x=x, y=y, color='yellow', marker='square', markersize=12)

    y = test_NSModel_step(func=caterpillar_quota_walk, params={'agent': None, 'Q1': 0.8, 'Q2': 0.5})
    plt.plot(x=x, y=y, color='green', marker='square', markersize=12)

    plt.title(label='Mean Time for Different NW Sampling Methods', size=12)
    plt.xlabel(xlabel='Number of Steps Executed by Agents', size=12)
    plt.ylabel(ylabel='Mean Time over 1000 Trials (ms)', size=12)
    plt.grid(b=True, which='major', axis='both')
    plt.show()

@pytest.mark.NSGridSearch
def test_NSGridSearch_zero_one() -> None:
    """Tests several rounds of NSGridSearch scored using 0/1 loss"""
    model = NSModel(method=NSMethod(func=random_walk, params={'agent': None}),
                    network=nx.complete_graph(n=10),
                    n_agents=10,
                    start_node=0)

    nsgs = NSGridSearch(model=model, target_func=random_walk,
                        target_func_params=None,
                        test_func=caterpillar_quota_walk,
                        test_func_params=None,
                        test_params={'Q1': 0.8, 'Q2': 0.5})
    best_params = nsgs.search(score_func=zero_one_loss, n_steps=10, n_jobs=5)
    print(best_params)

if __name__ == '__main__':
    # plot_NSModel_step_time()
    plot_NSMethod_time()

