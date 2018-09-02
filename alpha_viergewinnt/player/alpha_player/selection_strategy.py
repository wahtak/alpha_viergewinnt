import numpy as np


class SelectionStrategy():
    def __init__(self, exploration_factor):
        self.exploration_factor = exploration_factor

    def __call__(self, actions, attributes):
        return self.select_action(actions, attributes)

    def select_action(self, actions, attributes):
        raise NotImplementedError()

    def _get_potential_value(self, action_attributes):
        upper_confidence_bound = self.exploration_factor * (action_attributes.prior_probability /
                                                            (1 + action_attributes.visit_count))
        return action_attributes.action_value + upper_confidence_bound


class MaximumSelectionStrategy(SelectionStrategy):
    def __init__(self, exploration_factor):
        super().__init__(exploration_factor)

    def select_action(self, actions, attributes):
        potential_values = [self._get_potential_value(action_attributes) for action_attributes in attributes]
        return actions[np.argmax(potential_values)]


class SamplingSelectionStrategy(SelectionStrategy):
    def __init__(self, exploration_factor):
        super().__init__(exploration_factor)

    def select_action(self, actions, attributes):
        potential_values = np.array([self._get_potential_value(action_attributes) for action_attributes in attributes])
        probabilities = (potential_values + 1) / np.sum(potential_values + 1)
        # chose random index first, as list of actions can not always be correctly converted to numpy array
        selected_index = np.random.choice(np.arange(len(actions)), p=probabilities)
        return actions[selected_index]
