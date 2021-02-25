
#region
    # def optimize(self, graph: nx.Graph, sampling: NSMethod, scoring: NSMethod, n_jobs: int=1):
    #     """Chooses parameter values for the sampling algorithm that achieves an approximately optimal score when evaluating the network sample with the given scoring metric (with fixed parameter values). 

    #     Args:
    #         graph (nx.Graph): Network or graph to sample from
    #         sampling (NSMethod): Sampling algorithm with each value in 'params' being an iterable of values to grid search together
    #         scoring (NSMethod): Scoring metric with fixed constants as parameter values to evaluate sampling algorithm using each possible combination of testable parameter values in the algorithm
    #         n_jobs (int, optional): Number of jobs to divide optimization of sampling algorithm with multiprocessing. Defaults to 1.

    #     Returns:
    #         Tuple[Tuple, float, nx.Graph]: Best combination of parameter values for network sampling algorithm from the grid search; corresponsing best score; corresponding sample network or subgraph
    #     """
    #     # Setup recurrently used variables
    #     param_dict_keys = sampling.params.keys()
    #     param_combo_list = product(sampling.params.values())
    #     lock = Lock()

    #     # return values
    #     best_params = None
    #     best_score = 0
    #     sample = None

    #     # Sampling network in parallel
    #     # Block of code embedded as function to faciliate parallelization of grid search for network sampling algorithm parameters by testing single set of parameters
    #     def test_param_combo(param_combo: Tuple):
    #         """Scores the network sample for a single set of parameter values.

    #         Args:
    #             param_combo (Tuple): Combination of parameter values
    #         """
    #         # Access return values in outer function
    #         nonlocal best_params
    #         nonlocal best_score
    #         nonlocal sample

    #         ns = NetworkSampler(sampling=NSMethod(func=sampling.func, params={'graph': graph, **dict(zip(param_dict_keys[1:], param_combo))}), scoring=scoring)
    #         sample = ns.sample(graph=graph)
    #         score = ns.score()

    #         # Updating potential new best score and parameter combination
    #         lock.acquire()
    #         if score > best_score:
    #             best_score = score
    #             best_params = param_combo
    #         lock.release()

    #     with Parallel(n_jobs=n_jobs, backend='multiprocessing') as parallel_optimize:
    #         parallel_optimize()(delayed(test_param_combo)(param_combo) for param_combo in param_combo_list)
    #     return best_params, best_score, sample
#endregion
