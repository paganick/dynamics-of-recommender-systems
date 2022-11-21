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

    def initialize(self, initial_state: Opinion or SamplerOpinion or None = None, initialize_with_prejudice: bool = False) -> None:
        if initial_state is None:
            self.agent.initialize(initial_state=None,
                                  initialize_with_prejudice=initialize_with_prejudice)
        elif isinstance(initial_state, (OpinionType, SamplerOpinion)):
            self.agent.initialize(initial_state=initial_state,
                                  initialize_with_prejudice=initialize_with_prejudice)
        else:
            raise ValueError('Unknown input type.')
        self.algorithm.reset()

    def run(self, horizon: int, initialize: bool = True, initial_state: Opinion or SamplerOpinion or None = None) -> None:
        if initialize:
            self.initialize(initial_state=initial_state,
                            initialize_with_prejudice=False)
        reward = None
        for t in range(0, horizon):
            r = self.algorithm.compute_recommendation(reward=reward,
                                                      time=t)
            reward = self.agent.update_state(recommendation=r)

    def metrics(self) -> dict:  # TODO: metrics
        tol = 0.1
        # percentage
        delta_initial_opinion = np.abs(self.agent.trajectory['opinion'][0, :] - self.agent.trajectory['opinion'][-1, :])
        delta_initial_recommendation = np.abs(self.agent.trajectory['recommendation'][0, :] - self.agent.trajectory['opinion'][-1, :])
        # final distribution
        percentage = {'distance_initial_opinion': np.mean(delta_initial_opinion),
                      'distance_initial_recommendation': np.mean(delta_initial_recommendation),
                      'close_initial_opinion': np.sum(delta_initial_opinion < tol)/self.agent.n_agents(),
                      'close_initial_recommendation': np.sum(delta_initial_recommendation < tol)/self.agent.n_agents(),
                      'wasserstein_distance_initial_opinion': scipy.stats.wasserstein_distance(self.agent.trajectory['opinion'][0, :],
                                                                                               self.agent.trajectory['opinion'][-1, :]),
                      'wasserstein_distance_initial_recommendation': scipy.stats.wasserstein_distance(self.agent.trajectory['recommendation'][0, :],
                                                                                                      self.agent.trajectory['opinion'][-1, :]),
                      'final_distribution': self.agent.trajectory[KEY_OPINION][-1],
                      }
        return percentage

    def save(self) -> dict:  # TODO
        pass
