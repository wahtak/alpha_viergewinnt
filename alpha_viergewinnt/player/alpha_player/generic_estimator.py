import numpy as np


class GenericEstimator(object):
    def __init__(self, actions):
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
