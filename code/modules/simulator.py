from abc import ABC
from modules.agents import OpinionDynamicsEntity
from modules.algorithms import Algorithm
from modules.basic import Opinion, OpinionType
from modules.samplers import SamplerOpinion


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

    def metrics(self) -> None:  # TODO: metrics
        pass

    def save(self) -> dict:  # TODO
        pass
