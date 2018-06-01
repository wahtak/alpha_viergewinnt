from .attributes import ActionAttributes

from .selection_strategy import *


def test_call_selection_strategy():
    actions = [0, 1, 2]
    attributes = [
        ActionAttributes(action_value=0.1, prior_probability=1, visit_count=10),
        ActionAttributes(action_value=0.1, prior_probability=1, visit_count=1),
        ActionAttributes(action_value=0.0, prior_probability=0.1, visit_count=0)
    ]
    selection_strategy = SelectionStrategy(exploration_factor=1)

    assert selection_strategy(actions, attributes) == 1
