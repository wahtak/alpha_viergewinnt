import pytest
from copy import deepcopy
from collections import namedtuple

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Game, WinCondition, DrawCondition
from alpha_viergewinnt.player.alpha_player import AlphaTrainer, EvaluationModel
from alpha_viergewinnt.player.alpha_player.evaluation_model import VALUE_WIN, VALUE_LOSS, VALUE_PLAYER, VALUE_OPPONENT


@pytest.fixture
def game():
    return Game()


class DummyEstimator(object):
    KnowledgeEntry = namedtuple('KnowledgeEntry', ['state_array', 'selected_action', 'final_value'])

    def __init__(self):
        self.knowledge = []

    def infer(self, state_array):
        raise NotImplementedError()

    def learn(self, state_array, selected_action, final_value):
        knowledge_entry = DummyEstimator.KnowledgeEntry(state_array, selected_action, final_value)
        self.knowledge.append(knowledge_entry)
        dummy_loss = 0
        return dummy_loss


@pytest.fixture
def evaluation_models():
    win_condition_x = WinCondition(Player.X)
    win_condition_o = WinCondition(Player.O)
    draw_condition = DrawCondition()

    evaluation_model_x = EvaluationModel(
        estimator=DummyEstimator(),
        player=Player.X,
        opponent=Player.O,
        win_condition=win_condition_x,
        loss_condition=win_condition_o,
        draw_condition=draw_condition)

    evaluation_model_o = EvaluationModel(
        estimator=DummyEstimator(),
        player=Player.O,
        opponent=Player.X,
        win_condition=win_condition_o,
        loss_condition=win_condition_x,
        draw_condition=draw_condition)

    return evaluation_model_x, evaluation_model_o


def test_correct_learning_inputs(game, evaluation_models):
    evaluation_model_x, evaluation_model_o = evaluation_models
    trainers = {Player.X: AlphaTrainer(evaluation_model_x), Player.O: AlphaTrainer(evaluation_model_o)}

    def record_and_play(player, move):
        trainers[player].record(state=deepcopy(game), selected_action=move)
        game.play_move(player=player, move=move)

    record_and_play(player=Player.X, move=(0, 0))
    record_and_play(player=Player.O, move=(1, 0))
    record_and_play(player=Player.X, move=(1, 1))
    record_and_play(player=Player.O, move=(0, 1))
    record_and_play(player=Player.X, move=(2, 2))

    trainers[Player.X].learn(final_state=game)
    trainers[Player.O].learn(final_state=game)

    estimator_x = trainers[Player.X].evaluation_model.estimator
    estimator_o = trainers[Player.O].evaluation_model.estimator

    PL = VALUE_PLAYER
    OP = VALUE_OPPONENT

    assert len(estimator_x.knowledge) == 3

    assert estimator_x.knowledge[0].state_array.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert estimator_x.knowledge[0].selected_action == (0, 0)
    assert estimator_x.knowledge[0].final_value == VALUE_WIN

    assert estimator_x.knowledge[1].state_array.tolist() == [[PL, 0, 0], [OP, 0, 0], [0, 0, 0]]
    assert estimator_x.knowledge[1].selected_action == (1, 1)
    assert estimator_x.knowledge[1].final_value == VALUE_WIN

    assert estimator_x.knowledge[2].state_array.tolist() == [[PL, OP, 0], [OP, PL, 0], [0, 0, 0]]
    assert estimator_x.knowledge[2].selected_action == (2, 2)
    assert estimator_x.knowledge[2].final_value == VALUE_WIN

    assert len(estimator_o.knowledge) == 2

    assert estimator_o.knowledge[0].state_array.tolist() == [[OP, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert estimator_o.knowledge[0].selected_action == (1, 0)
    assert estimator_o.knowledge[0].final_value == VALUE_LOSS

    assert estimator_o.knowledge[1].state_array.tolist() == [[OP, 0, 0], [PL, OP, 0], [0, 0, 0]]
    assert estimator_o.knowledge[1].selected_action == (0, 1)
    assert estimator_o.knowledge[1].final_value == VALUE_LOSS
