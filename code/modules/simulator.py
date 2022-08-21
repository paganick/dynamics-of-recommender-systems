from abc import ABC
from modules.agents import OpinionDynamicsEntity
from modules.algorithms import Algorithm
from modules.basic import Opinion
from modules.samplers import SamplerOpinion


class Simulator(ABC):
    def __init__(self,
                 agent: OpinionDynamicsEntity,
                 algorithm: Algorithm,
                 initial_state: Opinion or None = None):
        super().__init__()
        self.agent = agent
        self.algorithm = algorithm
        self.initial_state = agent.initial_state()

    def initialize(self, initial_state: Opinion or SamplerOpinion or None = None, initialize_with_prejudice: bool = False) -> None:  # TODO: implement sampler
        if initial_state is None:
            self.agent.initialize(initial_state=self.initial_state,
                                  initialize_with_prejudice=initialize_with_prejudice)
        else:
            self.agent.initialize(initial_state=initial_state,
                                  initialize_with_prejudice=initialize_with_prejudice)
        self.algorithm.reset()

    def run(self, horizon: int, initialize: bool = True, initial_state: Opinion or SamplerOpinion or None = None) -> None:
        if initialize:
            self.initialize(initial_state=initial_state)
        reward = None
        for t in range(0, horizon):
            r = self.algorithm.compute_recommendation(reward=reward,
                                                      time=t)
            reward = self.agent.update_state(recommendation=r)

    def metrics(self) -> None:  # TODO: metrics
        pass 

    def save(self) -> dict:  # TODO
        pass
