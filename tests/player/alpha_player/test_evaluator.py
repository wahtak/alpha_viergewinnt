from collections import namedtuple

import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.player.alpha_player.evaluator import *


class DummyState(object):
    def __init__(self):
        self.board_size = (4, 4)
        self.state = 0
        self.winner = None
        self.draw = False

    def get_all_moves(self):
        return [0, 1, 2, 3]

    def get_possible_moves(self):
        return [0, 2, 3]

    def is_winner(self, player):
        return self.winner == player

    def is_draw(self):
        return self.draw

    def get_array_view(self, *args, **kwargs):
        return self.state


class DummyEstimator(object):
    KnowledgeEntry = namedtuple('KnowledgeEntry', ['state_array', 'target_distribution', 'target_state_value'])

    def __init__(self, actions):
        self.actions = actions
        self.knowledge = []

    def infer(self, state_array):
        uniform_prior_distribution = np.ones(len(self.actions)) / len(self.actions)
        state_value = 0.5
        return uniform_prior_distribution, state_value

    def train(self, state_array, target_distribution, target_state_value):
        knowledge_entry = DummyEstimator.KnowledgeEntry(state_array, target_distribution, target_state_value)
        self.knowledge.append(knowledge_entry)
        dummy_loss = 0
        return dummy_loss


@pytest.fixture
def state():
    return DummyState()


@pytest.fixture
def actions(state):
    return state.get_possible_moves()


@pytest.fixture
def evaluator(actions):
    estimator = DummyEstimator(actions=actions)
    return Evaluator(estimator=estimator, player=Player.X)


def test_evaluate_win_loss_draw(state, actions, evaluator):
    state.winner = Player.X
    prior_distribution, state_value, game_finished = evaluator.evaluate(state)

    assert len(prior_distribution) == len(actions)
    assert sum(prior_distribution) == pytest.approx(0)
    assert state_value == evaluator.STATE_VALUE_WIN
    assert game_finished is True


def test_evaluate_loss(state, actions, evaluator):
    state.winner = Player.O
    prior_distribution, state_value, game_finished = evaluator.evaluate(state)

    assert state_value == evaluator.STATE_VALUE_LOSS
    assert game_finished is True


def test_evaluate_draw(state, actions, evaluator):
    state.draw = True
    prior_distribution, state_value, game_finished = evaluator.evaluate(state)

    assert state_value == evaluator.STATE_VALUE_DRAW
    assert game_finished is True


def test_evaluate_not_win_loss_draw(state, actions, evaluator):
    prior_distribution, state_value, game_finished = evaluator.evaluate(state)

    assert len(prior_distribution) == len(actions)
    assert sum(prior_distribution) == pytest.approx(1)
    # from DummyEstimator
    assert state_value == 0.5
    assert game_finished is False


def test_train_when_finished(state, actions, evaluator):
    state.winner = Player.X
    search_distribution = np.array([0.1, 0.2, 0.7])
    evaluator.train(states_and_search_distributions=[(state, search_distribution)], final_state=state)

    # 1 batch with batch-size 1
    assert len(evaluator.estimator.knowledge) == 1
    assert len(evaluator.estimator.knowledge[0].state_array) == 1
    assert len(evaluator.estimator.knowledge[0].target_distribution) == 1
    assert len(evaluator.estimator.knowledge[0].target_state_value) == 1

    knowledge_batch = evaluator.estimator.knowledge[0]
    assert knowledge_batch.state_array[0] == state.get_array_view()
    assert np.all(knowledge_batch.target_distribution[0] == search_distribution)
    assert knowledge_batch.target_state_value[0] == evaluator.STATE_VALUE_WIN


def test_train_when_not_finished(state, actions, evaluator):
    with pytest.raises(GameNotFinishedException):
        evaluator.train(states_and_search_distributions=[], final_state=state)
