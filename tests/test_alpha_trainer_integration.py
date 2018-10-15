import pytest
from copy import deepcopy
from collections import namedtuple

import numpy as np

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.tictactoe import Game, WinCondition, DrawCondition
from alpha_viergewinnt.player.alpha_player import AlphaTrainer, Evaluator


@pytest.fixture
def game():
    return Game()


class DummyEstimator(object):
    STATE_VALUE_WIN = 1
    STATE_VALUE_LOSS = -1
    STATE_VALUE_DRAW = 0

    STATE_ARRAY_PLAYER = 1
    STATE_ARRAY_OPPONENT = -1

    KnowledgeEntry = namedtuple('KnowledgeEntry', ['state_array', 'target_distribution', 'target_state_value'])

    def __init__(self):
        self.knowledge = []

    def train(self, state_array, target_distribution, target_state_value):
        knowledge_entry = DummyEstimator.KnowledgeEntry(state_array, target_distribution, target_state_value)
        self.knowledge.append(knowledge_entry)
        dummy_loss = 0
        return dummy_loss


@pytest.fixture
def evaluators():
    win_condition_x = WinCondition(Player.X)
    win_condition_o = WinCondition(Player.O)
    draw_condition = DrawCondition()

    evaluator_x = Evaluator(
        estimator=DummyEstimator(),
        player=Player.X,
        opponent=Player.O,
        win_condition=win_condition_x,
        loss_condition=win_condition_o,
        draw_condition=draw_condition)

    evaluator_o = Evaluator(
        estimator=DummyEstimator(),
        player=Player.O,
        opponent=Player.X,
        win_condition=win_condition_o,
        loss_condition=win_condition_x,
        draw_condition=draw_condition)

    return evaluator_x, evaluator_o


def test_correct_training_inputs(game, evaluators):
    evaluator_x, evaluator_o = evaluators
    trainers = {
        Player.X: AlphaTrainer(evaluator_x, mcts_steps=None),
        Player.O: AlphaTrainer(evaluator_o, mcts_steps=None)
    }

    def record_and_play(player, move):
        search_distribution = np.zeros(len(game.get_all_moves()))
        search_distribution[move] = 1
        trainers[player]._record(state=deepcopy(game), search_distribution=search_distribution)
        game.play_move(player=player, move=move)

    record_and_play(player=Player.X, move=0)
    record_and_play(player=Player.O, move=3)
    record_and_play(player=Player.X, move=4)
    record_and_play(player=Player.O, move=1)
    record_and_play(player=Player.X, move=8)

    trainers[Player.X].train(final_state=game)
    trainers[Player.O].train(final_state=game)

    estimator_x = trainers[Player.X].evaluator.estimator
    estimator_o = trainers[Player.O].evaluator.estimator

    PL = estimator_x.STATE_ARRAY_PLAYER
    OP = estimator_x.STATE_ARRAY_OPPONENT

    # 1 batch with batch-size 3
    assert len(estimator_x.knowledge) == 1
    assert len(estimator_x.knowledge[0].state_array) == 3
    assert len(estimator_x.knowledge[0].target_distribution) == 3
    assert len(estimator_x.knowledge[0].target_state_value) == 3

    knowledge_batch = estimator_x.knowledge[0]
    assert knowledge_batch.state_array[0].tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert np.argmax(knowledge_batch.target_distribution[0]) == 0
    assert knowledge_batch.target_state_value[0] == estimator_x.STATE_VALUE_WIN

    assert knowledge_batch.state_array[1].tolist() == [[PL, 0, 0], [OP, 0, 0], [0, 0, 0]]
    assert np.argmax(knowledge_batch.target_distribution[1]) == 4
    assert knowledge_batch.target_state_value[1] == estimator_x.STATE_VALUE_WIN

    assert knowledge_batch.state_array[2].tolist() == [[PL, OP, 0], [OP, PL, 0], [0, 0, 0]]
    assert np.argmax(knowledge_batch.target_distribution[2]) == 8
    assert knowledge_batch.target_state_value[2] == estimator_x.STATE_VALUE_WIN

    PL = estimator_o.STATE_ARRAY_PLAYER
    OP = estimator_o.STATE_ARRAY_OPPONENT

    assert len(estimator_o.knowledge) == 1
    assert len(estimator_o.knowledge[0].state_array) == 2
    assert len(estimator_o.knowledge[0].target_distribution) == 2
    assert len(estimator_o.knowledge[0].target_state_value) == 2

    knowledge_batch = estimator_o.knowledge[0]
    assert knowledge_batch.state_array[0].tolist() == [[OP, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert np.argmax(knowledge_batch.target_distribution[0]) == 3
    assert knowledge_batch.target_state_value[0] == estimator_o.STATE_VALUE_LOSS

    assert knowledge_batch.state_array[1].tolist() == [[OP, 0, 0], [PL, OP, 0], [0, 0, 0]]
    assert np.argmax(knowledge_batch.target_distribution[1]) == 1
    assert knowledge_batch.target_state_value[1] == estimator_o.STATE_VALUE_LOSS
