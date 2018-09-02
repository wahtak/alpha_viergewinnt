from alpha_viergewinnt.player.alpha_player.attributes import ActionAttributes
from alpha_viergewinnt.player.alpha_player.selection_strategy import *


def test_call_selection_strategies():
    actions = [0, 1, 2]
    attributes = [
        ActionAttributes(action_value=0.1, prior_probability=1, visit_count=10),
        ActionAttributes(action_value=0.1, prior_probability=1, visit_count=1),
        ActionAttributes(action_value=0.0, prior_probability=0.1, visit_count=0)
    ]

    maximum_selection_strategy = MaximumSelectionStrategy(exploration_factor=1)
    assert maximum_selection_strategy(actions, attributes) == 1

    sampling_selection_strategy = SamplingSelectionStrategy(exploration_factor=1)
    assert sampling_selection_strategy(actions, attributes) in actions
