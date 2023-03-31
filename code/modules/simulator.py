from abc import ABC
import numpy as np
import scipy
from modules.agents import OpinionDynamicsEntity
from modules.algorithms import Algorithm
from modules.basic import Opinion, OpinionType
from modules.samplers import SamplerOpinion
from modules.utils import KEY_OPINION


class Simulator(ABC):
    def __init__(self,
                 agent: OpinionDynamicsEntity,
                 algorithm: Algorithm):
        super().__init__()
        self.agent = agent
        self.algorithm = algorithm

    def initialize(self,
                   initial_state: Opinion or SamplerOpinion or None = None,
                   initialize_with_prejudice: bool = False) -> None:
        if initial_state is None:
            self.agent.initialize(initial_state=None,
                                  initialize_with_prejudice=initialize_with_prejudice)
        elif isinstance(initial_state, (OpinionType, SamplerOpinion)):
            self.agent.initialize(initial_state=initial_state,
                                  initialize_with_prejudice=initialize_with_prejudice)
        else:
            raise ValueError('Unknown input type.')
        self.algorithm.reset()

    def run(self,
            horizon: int,
            initialize: bool = True,
            initial_state: Opinion or SamplerOpinion or None = None,
            initialize_with_prejudice: bool = False) -> None:
        if initialize:
            self.initialize(initial_state=initial_state,
                            initialize_with_prejudice=initialize_with_prejudice)
        reward = None
        for t in range(0, horizon):
            r = self.algorithm.compute_recommendation(reward=reward,
                                                      time=t)
            reward = self.agent.update_state(recommendation=r)

    def metrics(self) -> dict:  # TODO: metrics
        tol = 0.1
        # percentage
        final_opinions = self.agent.opinions()
        samples_recommendation = self.algorithm.recommendation_sampler[0].sample(self.agent.n_agents())
        samples_bias = self.agent.parameters().prejudice
        eta = self.agent.parameters().weight_prejudice/(1.0 - self.agent.parameters().weight_current_opinion)
        d_convergence_no_exploration = eta*samples_bias.reshape((1, -1)) + (1.0-eta)*samples_recommendation.reshape((-1, 1))
        d_convergence_no_exploration = d_convergence_no_exploration.reshape(-1)
        delta_initial_opinion = np.abs(self.agent.last_initial_state() - final_opinions)
        # delta_initial_recommendation = np.abs(samples_recommendation - self.agent.opinions())
        p = {'distance_initial_opinion': np.mean(delta_initial_opinion),
             'distance_initial_recommendation': 0.0,
             'close_initial_opinion': np.sum(delta_initial_opinion < tol)/self.agent.n_agents(),
             'close_initial_recommendation': 0.0,
             'wasserstein_distance_initial_opinion': scipy.stats.wasserstein_distance(self.agent.last_initial_state(),
                                                                                      final_opinions),
             'wasserstein_distance_initial_recommendation': scipy.stats.wasserstein_distance(samples_recommendation,
                                                                                             final_opinions),
             'wasserstein_distance_predicted_convergence': scipy.stats.wasserstein_distance(d_convergence_no_exploration,
                                                                                            final_opinions),
             'final_distribution': final_opinions,
             }
        return p

    def save(self) -> dict:  # TODO
        pass
