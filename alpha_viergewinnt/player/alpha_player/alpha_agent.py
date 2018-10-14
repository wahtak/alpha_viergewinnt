from copy import deepcopy

import numpy as np

from .graph import GameStateGraph, GameStatePath
from .mcts import Mcts


class AlphaAgent(object):
    def __init__(self, evaluation_model, mcts_steps, exploration_factor):
        self.evaluation_model = evaluation_model
        self.mcts_steps = mcts_steps
        self.exploration_factor = exploration_factor

    def get_next_move(self, state):
        return self._sample_action(self._get_search_probabilities(state))

    def _sample_action(self, search_probabilities):
        return np.random.choice(len(search_probabilities), p=search_probabilities)

    def _get_search_probabilities(self, state):
        # TODO: recycle graph from last call
        self.graph = GameStateGraph(state)
        mcts = Mcts(self.graph, GameStatePath, self.evaluation_model)

        for _ in range(self.mcts_steps):
            mcts.simulate_step(state)

        return mcts.get_search_probabilities(state, self.exploration_factor)

    def draw_graph(self):
        import matplotlib.pyplot as plt
        self.graph.draw()
        plt.show()


class AlphaPlayer(AlphaAgent):
    def __init__(self, evaluation_model, mcts_steps):
        super().__init__(evaluation_model, mcts_steps, exploration_factor=0.0)


class AlphaTrainer(AlphaAgent):
    def __init__(self, evaluation_model, mcts_steps):
        super().__init__(evaluation_model, mcts_steps, exploration_factor=1.0)
        self.states_and_selected_actions = []

    def get_next_move(self, state):
        selected_action = super().get_next_move(state)
        self._record(state, selected_action)
        return selected_action

    def _record(self, state, selected_action):
        self.states_and_selected_actions.append((deepcopy(state), selected_action))

    def learn(self, final_state):
        loss = self.evaluation_model.learn(self.states_and_selected_actions, final_state)
        self.states_and_selected_actions = []
        return loss