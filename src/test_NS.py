
from littleballoffur.exploration_sampling import (RandomWalkSampler,
                                                commonneighborawarerandomwalksampler,
                                                SnowBallSampler,
                                                CommunityStructureExpansionSampler,
                                                FrontierSampler)
import matplotlib.pyplot as plt
from mesa import Model
import networkx as nx
import numpy as np
import pytest
from timeit import timeit

from NetworkSampling import NSMethod, NetworkSampler, NetworkSamplerGrid
from NetworkSamplingFunctions import CaterpillarQuotaWalkSampler, RWSampler
from NetworkSamplingScoring import NetworkSamplerScorer

# def test_CaterpillarQuotaWalkSampler():
#     """[summary]
#     """
#     scorer = NSMethod(func=degree_sum, params=dict())
#     ns = NetworkSampler(sampler=CaterpillarQuotaWalkSampler(number_of_nodes=100, q1=0.00001, q2=0.00005), scorer=scorer)
#     sample = ns.sample(graph=graph, start_node=0)
#     nx.draw(G=sample, pos=nx.spring_layout(G=sample))
#     plt.show(block=False)
#     plt.savefig('./Graph Plots/CaterpillarQuotaWalkSampler.jpg', format='JPG')
#     print(f'Score: {ns.score()}')

if __name__ == '__main__':
    # test_RandomWalkSampler()
    # test_RWSampler()
    # test_CaterpillarQuotaWalkSampler()
    nsgrid = NetworkSamplerGrid(graph_group=[nx.path_graph(n)], 
                                )