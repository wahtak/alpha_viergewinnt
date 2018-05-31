from .graph import GameStateGraph, GameStatePath
from .mcts import Mcts


class AlphaPlayer(object):
    def __init__(self, selection_strategy, evaluation_model):
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model

    def get_next_move(self, state):
        pass
        # graph = GameStateGraph(state)
        # mcts = Mcts(graph, GameStatePath, self.selection_strategy, self.evaluation_model)
        # for _ in range(10):
        #     mcts.simulate_step()
        # mcts.get_action_probabilities()
