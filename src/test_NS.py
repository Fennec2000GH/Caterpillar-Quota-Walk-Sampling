
from littleballoffur.exploration_sampling import (RandomWalkSampler,
                                                CommonNeighborAwareRandomWalkSampler,
                                                SnowBallSampler,
                                                CommunityStructureExpansionSampler,
                                                FrontierSampler)
import copy
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples, qqline
from timeit import timeit
import tqdm
from typing import *

from NetworkSampling import NSMethod, NetworkSampler, NetworkSamplerTuner, NetworkSamplerGrid
from NetworkSamplingFunctions import CaterpillarQuotaWalkSampler, CaterpillarQuotaBFSSampler
from NetworkSamplingScorer import NetworkSamplingScorer

def connect_all_components(G: nx.Graph):
      """
      Connects each connected component of G with each other through a random;y chosen node in each component.

      Args:
            G (nx.Graph): Disconnected graph
      """
      print(f'number of connected components (before): {nx.number_connected_components(G=G1)}')

      # Gather a random node from each connected component
      component_nodes = list()
      for component in nx.connected_components(G=G1):
            rand_node = random.choice(seq=list(component))
            component_nodes.append(rand_node)

      # Connecting each selected node and adding edge to link together all other connected components
      for u, v in list(itertools.product(component_nodes, component_nodes)):
            if u != v:
                  G1.add_edge(u_of_edge=u, v_of_edge=v)

      print(f'number of connected components (after): {nx.number_connected_components(G=G1)}')

if __name__ == '__main__':
      g = nx.random_lobster(n=20, p1=0.5, p2=0)
      nx.draw(G=g, pos=nx.spring_layout(G=g))
      plt.show(block=False)
      plt.savefig(fname='./Output/caterpillar_tree.jpg', format='JPG', dpi=300)

      g = nx.random_lobster(n=20, p1=0.5, p2=0.25)
      nx.draw(G=g, pos=nx.spring_layout(G=g))
      plt.show(block=False)
      plt.savefig(fname='./Output/lobster_graph.jpg', format='JPG', dpi=300)

#       # Preliminary global data values used across all network samplings
#       start_node = 0
#       number_of_nodes = 100  # sample size
#       n_trials = 10  # Number of repeated scoring for each sampler and scoreing metric pair, which finalizes into a mean over all trials
#       pool = mp.Pool(processes=mp.cpu_count() * n_trials)

#       # Creates gnm random graphs and connects disconnected components together for each into strongly connected graphs
#       G1 = nx.Graph(incoming_graph_data=nx.random_graphs.gnm_random_graph(n=1000, m=5000), name='G1')
#       G2 = nx.Graph(incoming_graph_data=nx.random_graphs.gnm_random_graph(n=1000, m=50000), name='G2')
#       connect_all_components(G=G1)
#       connect_all_components(G=G2)
#       assert 0 in G1
#       assert 0 in G2
#       assert nx.number_connected_components(G=G1) == 1
#       assert nx.number_connected_components(G=G2) == 1

# #region
#       # Common tuning parameters
#       tuned_params = [None,
#                         None,
#                         dict({'k': np.arange(2, 21)}),
#                         None,
#                         dict({'q1':np.arange(0.0, 0.25, 0.01), 'q2':np.arange(0.25, 0.5, 0.01)})]

#       int_only = [False, False, True, False, False]

# #endregion

# #region
#       # Setting up network sampling grid

#       # Executing sampling grid to sample both graphs, G1 and G2, with each sampling algorithm and each scoring metric
#       sampler_group = [RandomWalkSampler(number_of_nodes=number_of_nodes),
#                                                 CommonNeighborAwareRandomWalkSampler(number_of_nodes=number_of_nodes),
#                                                 SnowBallSampler(number_of_nodes=number_of_nodes, k=2),
#                                                 CommunityStructureExpansionSampler(number_of_nodes=number_of_nodes),
#                                                 # FrontierSampler(number_of_nodes=number_of_nodes, number_of_seeds=1),
#                                                 CaterpillarQuotaWalkSampler(number_of_nodes=number_of_nodes, q1=0.1, q2=0.2)
#                                                 # CaterpillarQuotaBFSSampler(number_of_nodes=number_of_nodes, q1=q1_bfs_degree_sum_tuned)
#                                                 ]

#       sampler_names=['RandomWalkSampler',
#                         'CommonNeighborAwareRandomWalkSampler',
#                         'SnowBallSampler',
#                         'CommunityStructureExpansionSampler',
#                         # 'FrontierSampler',
#                         'CaterpillarQuotaWalkSampler'
#                         # 'CaterpillarQuotaBFSSampler'
#                         ]

#       nsgrid = NetworkSamplerGrid(graph_group=[G1,
#                                                 G2],
#                                     sampler_group=sampler_group,
#                                     sampler_names=sampler_names,
#                                     scorer_group=[NSMethod(func=NetworkSamplingScorer.degree_sum, params=dict()),
#                                                 NSMethod(func=NetworkSamplingScorer.distance_variance, params=dict({'start_node':start_node}))],
#                                     scorer_names=['degree_sum',
#                                                 'dist_var'],
#                                     dir_path='./Output',
#                                     csv_names=['G1_scores.csv',
#                                                 'G2_scores.csv'])

#       nsgrid.set_tuner(tuned_params=tuned_params, n_trials_tune=n_trials)
#       df_dict = nsgrid.sample_all_graphs(start_node=start_node, n_trials=n_trials)

# #endregion

# #region

#       # Degree distribution of samples by sampler for G1
#       degree_dist = dict()
#       transform = NSMethod(func=lambda n, graph: graph.degree[n], params=dict())
#       for sampler_name, sampler in zip(sampler_names, sampler_group):
#             print(f'sampler_name: {sampler_name}')
#             print(f'sampler: {sampler}')
#             sampler.__dict__.update(df_dict[G1].loc[sampler_name, 'degree_sum Tuned Params'])

#             samples = list()
#             try:
#                   samples = pool.starmap_async(sampler.sample, [(G1, start_node) for _ in np.arange(n_trials)]).get()
#             except:
#                   samples = [sampler.sample(G1, start_node) for _ in np.arange(n_trials)]

#             print(f'samples: {samples}')
#             dists = [NetworkSamplingScorer.distribution(sample, transform) for sample in samples]
#             print(f'dists: {dists}')

#             combined_dists = copy.deepcopy(x=dists[0])
#             combined_dists.clear()

#             for dist in dists:
#                   combined_dists.update(dist)

#             print(f'combined_dists: {combined_dists}')
#             degree_dist[sampler_name] = combined_dists

#       # Plotting degree density histograms
#       for sampler_name, combined_dists in degree_dist.items():
#             deg, deg_freq = list(combined_dists.keys()), list(combined_dists.values())
#             sum_deg_freq = np.sum(a=deg_freq, axis=None)
#             height = [f / sum_deg_freq for f in np.asarray(a=deg_freq).flatten()]

#             if __debug__:
#                   print(f'sampler_name: {sampler_name}')
#                   print(f'combined_dists: {combined_dists}')
#                   print(f'deg: {deg}')
#                   print(f'deg_freq: {deg_freq}')
#                   print(f'sum_deg_freq: {sum_deg_freq}')
#                   print(f'height: {height}')

#             fig, ax = plt.subplots()
#             bars = ax.bar(x=deg, height=height)
#             new_height = [bar.get_height() for bar in bars.patches]
#             plt.bar(x=deg, height=new_height, width=1.0, color='magenta')
#             pd.DataFrame(data=combined_dists.elements(), columns=['deg_freq']).deg_freq.plot.kde(color='green')
#             plt.title(label=f'{sampler_name} Degree Distribution (G1)')
#             plt.xlabel(xlabel='Degree')
#             plt.ylabel(ylabel='Relative Frequency')
#             plt.xticks(ticks=np.arange(0, 25, 1))
#             plt.xticks(fontsize=4, rotation=0)
#             plt.savefig(fname=f'./Output/{sampler_name}_G1_degree_distribution.jpg', format='JPG', dpi=300)

#       # Plotting facet grid of pairwise degree distributions subplot be subplot
#       fig, axes = plt.subplots(nrows=1, ncols=4, sharex='all', sharey='all', figsize=(12, 3))
#       fig.suptitle(t='2-Sample QQ Plots Relative to CaterpillarQuotaWalkSample (G1)\n', fontsize=12)
#       plt.tight_layout(pad=0.5)

#       for idx, sampler_name in enumerate(sampler_names):
#             if sampler_name.startswith('Caterpillar'):
#                   continue

#             pp_x = sm.ProbPlot(data=np.asarray(a=[deg for deg in degree_dist['CaterpillarQuotaWalkSampler'].values()]))
#             pp_y = sm.ProbPlot(data=np.asarray(a=[deg for deg in degree_dist[sampler_name].values()]))
#             qqplot_2samples(pp_x, pp_y, ax=axes[idx])
#             qqline(ax=axes[idx], line='45')
#             axes[idx].set_title(label=sampler_name, fontsize=8)
#             axes[idx].set_xlabel(xlabel=None)
#             axes[idx].set_ylabel(ylabel=None)

#       # plt.xlabel(xlabel='CQWS Degree')
#       # plt.ylabel(ylabel='Second Sampler\'s Degree')
#       plt.show(block=False)
#       plt.savefig(fname=f'./Output/{sampler_name}_G1_degree_facet.jpg', format='JPG', dpi=300)

# #endregion

# #region

#       # Degree distribution of samples by sampler for G2
#       degree_dist = dict()
#       transform = NSMethod(func=lambda n, graph: graph.degree[n], params=dict())
#       for sampler_name, sampler in zip(sampler_names, sampler_group):
#             print(f'sampler_name: {sampler_name}')
#             print(f'sampler: {sampler}')
#             sampler.__dict__.update(df_dict[G2].loc[sampler_name, 'degree_sum Tuned Params'])

#             samples = list()
#             try:
#                   samples = pool.starmap_async(sampler.sample, [(G2, start_node) for _ in np.arange(n_trials)]).get()
#             except:
#                   samples = [sampler.sample(G2, start_node) for _ in np.arange(n_trials)]

#             print(f'samples: {samples}')
#             dists = [NetworkSamplingScorer.distribution(sample, transform) for sample in samples]
#             print(f'dists: {dists}')

#             combined_dists = copy.deepcopy(x=dists[0])
#             combined_dists.clear()

#             for dist in dists:
#                   combined_dists.update(dist)

#             print(f'combined_dists: {combined_dists}')
#             degree_dist[sampler_name] = combined_dists

#       # Plotting degree density histograms
#       for sampler_name, combined_dists in degree_dist.items():
#             deg, deg_freq = list(combined_dists.keys()), list(combined_dists.values())
#             sum_deg_freq = np.sum(a=deg_freq, axis=None)
#             height = [f / sum_deg_freq for f in np.asarray(a=deg_freq).flatten()]

#             if __debug__:
#                   print(f'sampler_name: {sampler_name}')
#                   print(f'combined_dists: {combined_dists}')
#                   print(f'deg: {deg}')
#                   print(f'deg_freq: {deg_freq}')
#                   print(f'sum_deg_freq: {sum_deg_freq}')
#                   print(f'height: {height}')

#             fig, ax = plt.subplots()
#             bars = ax.bar(x=deg, height=height)
#             new_height = [bar.get_height() for bar in bars.patches]
#             plt.bar(x=deg, height=new_height, width=1.0, color='magenta')
#             pd.DataFrame(data=combined_dists.elements(), columns=['deg_freq']).deg_freq.plot.kde(color='green')
#             plt.title(label=f'{sampler_name} Degree Distribution (G2)')
#             plt.xlabel(xlabel='Degree')
#             plt.ylabel(ylabel='Relative Frequency')
#             plt.xticks(ticks=np.arange(0, 25, 1))
#             plt.xticks(fontsize=4, rotation=0)
#             plt.savefig(fname=f'./Output/{sampler_name}_G2_degree_distribution.jpg', format='JPG', dpi=300)

#       # Plotting facet grid of pairwise degree distributions subplot be subplot
#       fig, axes = plt.subplots(nrows=1, ncols=4, sharex='all', sharey='all', figsize=(12, 3))
#       fig.suptitle(t='2-Sample QQ Plots Relative to CaterpillarQuotaWalkSample (G2)\n', fontsize=12)
#       plt.tight_layout(pad=0.5)

#       for idx, sampler_name in enumerate(sampler_names):
#             if sampler_name.startswith('Caterpillar'):
#                   continue

#             pp_x = sm.ProbPlot(data=np.asarray(a=[deg for deg in degree_dist['CaterpillarQuotaWalkSampler'].values()]))
#             pp_y = sm.ProbPlot(data=np.asarray(a=[deg for deg in degree_dist[sampler_name].values()]))
#             qqplot_2samples(pp_x, pp_y, ax=axes[idx])
#             qqline(ax=axes[idx], line='45')
#             axes[idx].set_title(label=sampler_name, fontsize=8)
#             axes[idx].set_xlabel(xlabel=None)
#             axes[idx].set_ylabel(ylabel=None)

#       # plt.xlabel(xlabel='CQWS Degree Quantiles')
#       # plt.ylabel(ylabel='Second Sampler\'s Degree Quantiles')
#       plt.show(block=False)
#       plt.savefig(fname=f'./Output/{sampler_name}_G2_degree_facet.jpg', format='JPG', dpi=300)

# #endregion

#       # Printing tuned parameters geared towards performing better at either degree sum or distance variance
#       # Only SnowballSampler, CaterpillarQuotaWalkSampler, and CaterpillarBFSSampler have tunable parameters

#       print(f'df_dict:\n{df_dict}')
