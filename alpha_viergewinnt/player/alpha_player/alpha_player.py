import numpy as np

from .graph import GameStateGraph, GameStatePath
from .mcts import Mcts


class AlphaPlayer(object):
    def __init__(self, evaluation_model, mcts_steps, exploration_factor=1):
        self.evaluation_model = evaluation_model
        self.mcts_steps = mcts_steps
        self.exploration_factor = exploration_factor

    def get_next_move(self, state):
        # TODO: recycle graph from last call
        self.graph = GameStateGraph(state)
        mcts = Mcts(self.graph, GameStatePath, self.evaluation_model)

        for _ in range(self.mcts_steps):
            mcts.simulate_step(state)

        search_probabilities = mcts.get_search_probabilities(state, self.exploration_factor)
        return self._sample_action(search_probabilities)

    def _sample_action(self, search_probabilities):
        return np.random.choice(len(search_probabilities), p=search_probabilities)

    def draw_graph(self):
        import matplotlib.pyplot as plt
        self.graph.draw()
        plt.show()
