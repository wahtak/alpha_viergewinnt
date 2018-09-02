import numpy as np


class GenericEstimator(object):
    def __init__(self, board_size, actions, **kwargs):
        self.actions = actions

    def infer(self, state_array):
        # dummy estimation
        uniform_action_values = np.ones(len(self.actions)) / len(self.actions)
        return uniform_action_values, 0

    def learn(self, state_array, selected_action, final_value):
        # dummy learning
        dummy_loss = 0
        return dummy_loss

    def load(self):
        pass

    def save(self):
        pass
