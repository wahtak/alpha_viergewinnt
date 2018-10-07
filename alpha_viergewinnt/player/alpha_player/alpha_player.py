from .graph import GameStateGraph, GameStatePath
from .mcts import Mcts


class AlphaPlayer(object):
    def __init__(self, selection_strategy, evaluation_model, mcts_steps):
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model
        self.mcts_steps = mcts_steps

    def get_next_move(self, state):
        # TODO: recycle graph from last call
        self.graph = GameStateGraph(state)
        mcts = Mcts(self.graph, GameStatePath, self.selection_strategy, self.evaluation_model)

        for _ in range(self.mcts_steps):
            mcts.simulate_step(state)

        return self._select_action(state)

    def _select_action(self, state):
        actions = self.graph.get_actions(state)
        attributes = self.graph.get_attributes(state)
        selected_action = self.selection_strategy(actions, attributes)
        return selected_action

    def draw_graph(self):
        self.graph.draw()
