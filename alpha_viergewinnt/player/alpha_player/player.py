from .graph import GameStateGraph, GameStatePath
from .mcts import Mcts


class AlphaPlayer(object):
    def __init__(self, selection_strategy, evaluation_model):
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model

    def get_next_move(self, state):
        # TODO: recycle graph from last call
        self.graph = GameStateGraph(state)
        mcts = Mcts(self.graph, GameStatePath, self.selection_strategy, self.evaluation_model)

        for _ in range(10):
            mcts.simulate_step(state)

        return self._select_action(state)

    def _select_action(self, state):
        actions = self.graph.get_actions(state)
        attributes = [self.graph.get_action_attributes(state, action) for action in actions]
        selected_action = self.selection_strategy(actions, attributes)
        return selected_action
