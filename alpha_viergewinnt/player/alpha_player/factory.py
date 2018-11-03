from .generic_estimator import GenericEstimator
from .mlp_estimator import MlpEstimator
from .evaluator import Evaluator
from .alpha_agent import AlphaPlayer, AlphaTrainer


def create_generic_estimator(game):
    return GenericEstimator(actions=game.get_all_moves())


def create_mlp_estimator(game):
    return MlpEstimator(board_size=game.board_size, actions=game.get_all_moves())


def create_alpha_player(estimator, player, mcts_steps, exploration_factor=0.1, random_seed=None):
    evaluator = Evaluator(estimator, player)
    return AlphaPlayer(evaluator, mcts_steps, exploration_factor, random_seed)


def create_alpha_trainer(estimator, player, mcts_steps, exploration_factor=1.0, random_seed=None):
    evaluator = Evaluator(estimator, player)
    return AlphaTrainer(evaluator, mcts_steps, exploration_factor, random_seed)
