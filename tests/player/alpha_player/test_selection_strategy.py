import numpy as np

from alpha_viergewinnt.player.alpha_player.attributes import Attributes
from alpha_viergewinnt.player.alpha_player.selection_strategy import *


def test_call_selection_strategies():
    actions = [0, 1, 3]
    attributes = Attributes(state_value=None, prior_probabilities=np.array([1, 1, 1, 0.1]))
    attributes.action_values = np.array([0.1, 0.1, 0.2, 0])
    attributes.visit_counts = np.array([10, 1, 0, 0])

    maximum_selection_strategy = MaximumSelectionStrategy(exploration_factor=1)
    assert maximum_selection_strategy(actions, attributes) == 1

    sampling_selection_strategy = SamplingSelectionStrategy(exploration_factor=1)
    assert sampling_selection_strategy(actions, attributes) in actions
