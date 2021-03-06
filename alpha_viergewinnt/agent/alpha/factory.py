from .generic_estimator import GenericEstimator
from .mlp_estimator import MlpEstimator
from .evaluator import Evaluator
from .alpha import AlphaAgent, AlphaTrainer


def create_generic_estimator(game):
    return GenericEstimator(actions=game.get_all_moves())


def create_mlp_estimator(game):
    return MlpEstimator(board_size=game.board_size, actions=game.get_all_moves())


def create_alpha_agent(estimator, player, mcts_steps, random_seed=None, draw_graph=False):
    evaluator = Evaluator(estimator, player)
    return AlphaAgent(evaluator=evaluator, mcts_steps=mcts_steps, random_seed=random_seed, draw_graph=draw_graph)


def create_alpha_trainer(estimator, player, mcts_steps, random_seed=None, draw_graph=False):
    evaluator = Evaluator(estimator, player)
    return AlphaTrainer(evaluator, mcts_steps, random_seed, draw_graph)
