import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.player.alpha_player.evaluation_model import *


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

    def __init__(self, actions):
        self.actions = actions
        self.knowledge = []

    def infer(self, state_array):
        uniform_prior_distribution = np.ones(len(self.actions)) / len(self.actions)
        state_value = 0.5
        return uniform_prior_distribution, state_value

    def learn(self, state_array, selected_action, final_state_value):
        self.knowledge.append((state_array, selected_action, final_state_value))
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
    evaluation_model = EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    prior_distribution, state_value, game_finished = evaluation_model(actions, state)

    assert len(prior_distribution) == len(actions)
    assert sum(prior_distribution) == pytest.approx(0)
    assert state_value == estimator.STATE_VALUE_WIN
    assert game_finished is True

    # loss
    evaluation_model = EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=true_condition,
        draw_condition=false_condition)
    prior_distribution, state_value, game_finished = evaluation_model(actions, state)

    assert state_value == estimator.STATE_VALUE_LOSS
    assert game_finished is True

    # draw
    evaluation_model = EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=true_condition)
    prior_distribution, state_value, game_finished = evaluation_model(actions, state)

    assert state_value == estimator.STATE_VALUE_DRAW
    assert game_finished is True


def test_evaluate_not_win_loss_draw(state, actions, estimator, true_condition, false_condition):
    evaluation_model = EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    prior_distribution, state_value, game_finished = evaluation_model(actions, state)

    assert len(prior_distribution) == len(actions)
    assert sum(prior_distribution) == pytest.approx(1)
    # from DummyEstimator
    assert state_value == 0.5
    assert game_finished is False


def test_learn_when_finished(state, actions, estimator, true_condition, false_condition):
    # game finished with win
    evaluation_model = EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    selected_action = 0
    evaluation_model.learn(states_and_selected_actions=[(state, selected_action)], final_state=state)

    assert estimator.knowledge == [(state.get_array_view(), selected_action, estimator.STATE_VALUE_WIN)]


def test_learn_when_not_finished(state, actions, estimator, true_condition, false_condition):
    # game not finished
    evaluation_model = EvaluationModel(
        estimator=estimator,
        player=Player.X,
        opponent=Player.O,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)

    with pytest.raises(GameNotFinishedException):
        evaluation_model.learn(states_and_selected_actions=[], final_state=state)
