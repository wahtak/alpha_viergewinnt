import pytest

from alpha_viergewinnt.player.alpha_player.evaluation_model import *


class DummyState(object):
    def __init__(self):
        self.board_size = (4, 4)
        self.state = np.zeros(shape=self.board_size, dtype=np.int16)

    def get_all_moves(self):
        return [0, 1, 2, 3]

    def get_possible_moves(self):
        return [0, 2, 3]

    def check(self, condition):
        return condition.check(self)


class DummyEstimator(object):
    def __init__(self, actions):
        self.actions = actions
        self.knowledge = []

    def infer(self, state):
        uniform_prior_probabilities = np.ones(len(self.actions)) / len(self.actions)
        state_value = 0.5
        return uniform_prior_probabilities, state_value

    def learn(self, state, selected_action, value):
        self.knowledge.append((state, selected_action, value))


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
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert len(prior_probabilities) == len(actions)
    assert sum(prior_probabilities) == pytest.approx(0)
    assert state_value == VALUE_WIN

    # loss
    evaluation_model = EvaluationModel(
        estimator=estimator,
        win_condition=false_condition,
        loss_condition=true_condition,
        draw_condition=false_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert state_value == VALUE_LOSS

    # draw
    evaluation_model = EvaluationModel(
        estimator=estimator,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=true_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert state_value == VALUE_DRAW


def test_evaluate_not_win_loss_draw(state, actions, estimator, true_condition, false_condition):
    evaluation_model = EvaluationModel(
        estimator=estimator,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    prior_probabilities, state_value = evaluation_model(actions, state)

    assert len(prior_probabilities) == len(actions)
    assert sum(prior_probabilities) == pytest.approx(1)
    # from DummyEstimator
    assert state_value == 0.5


def test_learn_when_finished(state, actions, estimator, true_condition, false_condition):
    # game finished with win
    evaluation_model = EvaluationModel(
        estimator=estimator,
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)
    selected_action = 0
    evaluation_model.learn(states_and_selected_actions=[(state, selected_action)], final_state=state)

    assert estimator.knowledge == [(state, selected_action, VALUE_WIN)]


def test_learn_when_not_finished(state, actions, estimator, true_condition, false_condition):
    # game not finished
    evaluation_model = EvaluationModel(
        estimator=estimator,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=false_condition)

    with pytest.raises(GameNotFinishedException):
        evaluation_model.learn(states_and_selected_actions=[], final_state=state)
