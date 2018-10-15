from collections import namedtuple

import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.player.alpha_player.evaluator import *


class DummyState(object):
    def __init__(self):
        self.board_size = (4, 4)
        self.state = 0

    def get_all_moves(self):
        return [0, 1, 2, 3]

    def get_possible_moves(self):
        return [0, 2, 3]

    def check(self, condition):
        return condition.check(self)

    def get_array_view(self, *args, **kwargs):
        return self.state


class DummyEstimator(object):
    # constants for state values and state array
    STATE_VALUE_WIN = 1
    STATE_VALUE_LOSS = -1
    STATE_VALUE_DRAW = 0

    STATE_ARRAY_PLAYER = 1
    STATE_ARRAY_OPPONENT = -1

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
def false_condition():
    class FalseCondition(object):
        def check(self, _):
            return False

    return FalseCondition()


@pytest.fixture
def true_condition():
    class TrueCondition(object):
        def check(self, _):
            return True

    return TrueCondition()


@pytest.fixture
def state():
    return DummyState()


@pytest.fixture
def actions(state):
    return state.get_possible_moves()


@pytest.fixture
def estimator(actions):
    return DummyEstimator(actions=actions)


def test_evaluate_win_loss_draw(state, actions, estimator, true_condition, false_condition):
    # win
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    prior_distribution, state_value, game_finished = evaluator(state)

    assert len(prior_distribution) == len(actions)
    assert sum(prior_distribution) == pytest.approx(0)
    assert state_value == estimator.STATE_VALUE_WIN
    assert game_finished is True

    # loss
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=true_condition,
        draw_condition=false_condition)
    prior_distribution, state_value, game_finished = evaluator(state)

    assert state_value == estimator.STATE_VALUE_LOSS
    assert game_finished is True

    # draw
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=true_condition)
    prior_distribution, state_value, game_finished = evaluator(state)

    assert state_value == estimator.STATE_VALUE_DRAW
    assert game_finished is True


def test_evaluate_not_win_loss_draw(state, actions, estimator, true_condition, false_condition):
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    prior_distribution, state_value, game_finished = evaluator(state)

    assert len(prior_distribution) == len(actions)
    assert sum(prior_distribution) == pytest.approx(1)
    # from DummyEstimator
    assert state_value == 0.5
    assert game_finished is False


def test_train_when_finished(state, actions, estimator, true_condition, false_condition):
    # game finished with win
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    search_distribution = np.array([0.1, 0.2, 0.7])
    evaluator.train(states_and_search_distributions=[(state, search_distribution)], final_state=state)

    # 1 batch with batch-size 1
    assert len(estimator.knowledge) == 1
    assert len(estimator.knowledge[0].state_array) == 1
    assert len(estimator.knowledge[0].target_distribution) == 1
    assert len(estimator.knowledge[0].target_state_value) == 1

    knowledge_batch = estimator.knowledge[0]
    assert knowledge_batch.state_array[0] == state.get_array_view()
    assert np.all(knowledge_batch.target_distribution[0] == search_distribution)
    assert knowledge_batch.target_state_value[0] == estimator.STATE_VALUE_WIN


def test_train_when_not_finished(state, actions, estimator, true_condition, false_condition):
    # game not finished
    evaluator = Evaluator(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)

    with pytest.raises(GameNotFinishedException):
        evaluator.train(states_and_search_distributions=[], final_state=state)
