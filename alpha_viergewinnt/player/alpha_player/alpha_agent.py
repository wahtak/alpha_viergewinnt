from copy import deepcopy

import numpy as np

from .graph import GameStateGraph
from .mcts import Mcts


class AlphaAgent(object):
    def __init__(self, evaluator, mcts_steps, random_seed, exploration_factor):
        self.evaluator = evaluator
        self.mcts_steps = mcts_steps
        self.exploration_factor = exploration_factor
        self._random_state = np.random.RandomState(random_seed)

    def get_next_move(self, state):
        return self._sample_action(self._get_search_distribution(state))

    def _sample_action(self, search_distribution):
        return self._random_state.choice(len(search_distribution), p=search_distribution)

    def _get_search_distribution(self, state):
        # TODO: recycle graph from last call
        self.graph = GameStateGraph(state)
        mcts = Mcts(self.graph, self.evaluator)

        for _ in range(self.mcts_steps):
            mcts.simulate_step(state)

        return mcts.get_search_distribution(state, self.exploration_factor)

    def draw_graph(self):
        self.graph.draw()


class AlphaPlayer(AlphaAgent):
    def __init__(self, evaluator, mcts_steps, random_seed=None):
        super().__init__(evaluator, mcts_steps, random_seed, exploration_factor=0.1)


class AlphaTrainer(AlphaAgent):
    def __init__(self, evaluator, mcts_steps, random_seed=None):
        super().__init__(evaluator, mcts_steps, random_seed, exploration_factor=1.0)
        self.states_and_search_distributions = []

    def get_next_move(self, state):
        search_distribution = self._get_search_distribution(state)
        self._record(state, search_distribution)
        selected_action = self._sample_action(search_distribution)
        return selected_action

    def _record(self, state, search_distribution):
        self.states_and_search_distributions.append((deepcopy(state), search_distribution))

    def train(self, final_state):
        loss = self.evaluator.train(self.states_and_search_distributions, final_state)
        self.states_and_search_distributions = []
        return loss
