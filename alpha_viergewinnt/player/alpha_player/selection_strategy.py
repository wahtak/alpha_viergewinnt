import numpy as np


class SelectionStrategy():
    def __init__(self, exploration_factor):
        self.exploration_factor = exploration_factor

    def __call__(self, actions, attributes):
        return self.select_action(actions, attributes)

    def select_action(self, actions, attributes):
        raise NotImplementedError()

    def _get_potential_values(self, actions, attributes):
        upper_confidence_bound = self.exploration_factor * (attributes.prior_probabilities /
                                                            (1 + attributes.visit_counts))
        all_potential_values = attributes.action_values + upper_confidence_bound
        return self._mask_values(all_potential_values, actions)

    def _mask_values(self, values, actions):
        mask = np.zeros_like(values)
        mask[actions] = 1
        return values * mask


class MaximumSelectionStrategy(SelectionStrategy):
    def __init__(self, exploration_factor):
        super().__init__(exploration_factor)

    def select_action(self, actions, attributes):
        return np.argmax(self._get_potential_values(actions, attributes))


class SamplingSelectionStrategy(SelectionStrategy):
    def __init__(self, exploration_factor):
        super().__init__(exploration_factor)

    def select_action(self, actions, attributes):
        potential_values = self._get_potential_values(actions, attributes)
        probabilities = (potential_values + 1) / np.sum(potential_values + 1)
        # chose random index first, as list of actions can not always be correctly converted to numpy array
        return np.random.choice(np.arange(len(probabilities)), p=probabilities)
