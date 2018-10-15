import numpy as np


class GenericEstimator(object):
    # constants for state values and state array
    STATE_VALUE_WIN = 1
    STATE_VALUE_LOSS = -1
    STATE_VALUE_DRAW = 0

    STATE_ARRAY_PLAYER = 1
    STATE_ARRAY_OPPONENT = -1

    def __init__(self, board_size, actions, **kwargs):
        self.actions = actions

    def infer(self, state_array):
        # dummy estimation
        uniform_action_value = np.ones(len(self.actions)) / len(self.actions)
        return uniform_action_value, 0

    def train(self, state_array, target_distribution, target_state_value):
        # dummy training
        dummy_loss = 0
        return dummy_loss

    def load(self):
        pass

    def save(self):
        pass
