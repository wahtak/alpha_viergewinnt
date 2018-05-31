import pytest

from .test_mcts import DummyState, select_first_strategy, max_first_model
from .player import *


@pytest.mark.skip
def test_get_next_move():
    player = AlphaPlayer(select_first_strategy, max_first_model)
    root = DummyState()
    action = player.get_next_move(root)

    assert action == root.get_possible_moves()[0]
