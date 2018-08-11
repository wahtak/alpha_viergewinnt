import numpy as np


class GenericEstimator(object):
    def __init__(self, board_size, actions):
        self.actions = actions

    def infer(self, state):
        # dummy estimation
        uniform_action_values = np.ones(len(self.actions)) / len(self.actions)
        return uniform_action_values, 0

    def learn(self, state, selected_action, final_value):
        # dummy learning
        dummy_loss = 0
        return dummy_loss
