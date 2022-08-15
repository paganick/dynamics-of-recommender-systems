from abc import ABC
from modules.agents import OpinionDynamicsEntity
from modules.algorithms import Algorithm


class Simulator(ABC):
    def __init__(self,
                 agent: OpinionDynamicsEntity,
                 algorithm: Algorithm):
        super().__init__()
        self.agent = agent
        self.algorithm = algorithm

    def run(self, horizon: int) -> None:
        reward = None
        for t in range(0, horizon):
            r = self.algorithm.compute_recommendation(reward=reward,
                                                      time=t)
            reward = self.agent.update_state(recommendation=r)

    def metrics(self) -> None: # TODO: metrics
        pass

    def save(self) -> dict:  # TODO
        pass
