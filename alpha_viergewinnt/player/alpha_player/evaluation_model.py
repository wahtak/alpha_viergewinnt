import numpy as np

STATE_VALUE_WIN = 1
STATE_VALUE_LOSS = -1
STATE_VALUE_DRAW = 0


class EvaluationModel(object):
    def __init__(self, board_size, win_condition, loss_condition, draw_condition):
        self.win_condition = win_condition
        self.loss_condition = loss_condition
        self.draw_condition = draw_condition

    def __call__(self, actions, state):
        return self.get_prior_probabilities_and_state_value(actions, state)

    def get_prior_probabilities_and_state_value(self, actions, state):
        uniform_action_priors = np.ones(len(actions)) / len(actions)

        if state.check(self.win_condition):
            return uniform_action_priors, STATE_VALUE_WIN

        elif state.check(self.loss_condition):
            return uniform_action_priors, STATE_VALUE_LOSS

        elif state.check(self.draw_condition):
            return uniform_action_priors, STATE_VALUE_DRAW

        else:
            return self.estimate_prior_probabilities_and_state_value(actions, state)

    def estimate_prior_probabilities_and_state_value(self, actions, state):
        # dummy estimation
        uniform_action_priors = np.ones(len(actions)) / len(actions)
        return uniform_action_priors, STATE_VALUE_DRAW
