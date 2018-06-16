import pytest

from alpha_viergewinnt.player.alpha_player.evaluation_model import *


class DummyState(object):
    def check(self, condition):
        return condition.check(self)


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
def dummy_state_and_actions():
    state = DummyState()
    actions = ['a', 'b', 'c']

    return state, actions


def test_get_prior_probabilities_and_state_value(dummy_state_and_actions, true_condition, false_condition):
    state, actions = dummy_state_and_actions

    # win
    evaluation_model = ConditionEvaluationModel(
        win_condition=true_condition, loss_condition=false_condition, draw_condition=false_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert len(prior_probabilities) == len(actions)
    assert sum(prior_probabilities) == pytest.approx(1)
    assert state_value == 1

    # loss
    evaluation_model = ConditionEvaluationModel(
        win_condition=false_condition, loss_condition=true_condition, draw_condition=false_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert state_value == -1

    # draw
    evaluation_model = ConditionEvaluationModel(
        win_condition=false_condition, loss_condition=false_condition, draw_condition=true_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert state_value == 0
