from copy import deepcopy
import logging

import numpy as np

from .graph import GameStateGraph
from .mcts import Mcts


class Alpha(object):
    def __init__(self, evaluator, mcts_steps, random_seed, draw_graph):
        self.logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)
        self.evaluator = evaluator
        self.mcts_steps = mcts_steps
        self.random_state = np.random.RandomState(random_seed)
        self.graph = None
        self.draw_graph = draw_graph

    def _sample_action(self, search_distribution):
        return self.random_state.choice(len(search_distribution), p=search_distribution)

    def _get_search_distribution(self, state, exploration_factor=1.0):
        # TODO: only reset root and keep rest of graph
        self.graph = GameStateGraph(state)
        mcts = Mcts(self.graph, self.evaluator)

        for _ in range(self.mcts_steps):
            mcts.simulate_step(state)

        search_distribution = mcts.get_search_distribution(state, exploration_factor)

        self.logger.debug("mean node depth: %.2f" % self.graph.get_mean_node_depth())
        self.logger.debug("search distribution: %s" % search_distribution)
        if self.draw_graph:
            self.graph.draw()

        return search_distribution


class AlphaAgent(Alpha):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_next_move(self, state):
        return self._sample_action(self._get_search_distribution(state))


class AlphaTrainer(Alpha):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states_and_search_distributions = []
        self.move_count = 0

    def get_next_move(self, state):
        self.move_count += 1
        exploration_factor = max(0.1, 1.0 / self.move_count)
        search_distribution = self._get_search_distribution(state, exploration_factor)
        self._record(state, search_distribution)
        selected_action = self._sample_action(search_distribution)
        return selected_action

    def _record(self, state, search_distribution):
        self.states_and_search_distributions.append((deepcopy(state), search_distribution))

    def train(self, final_state):
        loss = self.evaluator.train(self.states_and_search_distributions, final_state)
        self.states_and_search_distributions = []
        return loss
