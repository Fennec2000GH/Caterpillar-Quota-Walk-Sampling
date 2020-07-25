
import networkx as nx
import test
from NetworkSampling import NSAgent, NSModel, NSMethod
from NetworkSamplingFunctions import random_walk, caterpillar_quota_walk
from NetworkSamplingMetrics import NSGridSearch
from NetworkSamplingScoring import zero_one_loss

# @test.mark.NSGridSearch
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
    test_NSGridSearch_zero_one()

