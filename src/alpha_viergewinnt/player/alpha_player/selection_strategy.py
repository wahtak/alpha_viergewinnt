import numpy as np


class SelectionStrategy():
    def __init__(self, exploration_factor):
        self.exploration_factor = exploration_factor

    def __call__(self, actions, attributes):
        potential_values = [self._get_potential_value(action_attributes) for action_attributes in attributes]
        return actions[np.argmax(potential_values)]

    def _get_potential_value(self, action_attributes):
        upper_confidence_bound = self.exploration_factor * (action_attributes.prior_probability /
                                                            (1 + action_attributes.visit_count))
        return action_attributes.action_value + upper_confidence_bound
