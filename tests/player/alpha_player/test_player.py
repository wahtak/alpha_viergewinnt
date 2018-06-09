import pytest

from alpha_viergewinnt.player.alpha_player.player import *

from .test_mcts import DummyState
from .test_mcts import select_first_strategy as select_first_strategy_
from .test_mcts import max_first_model as max_first_model_


@pytest.fixture
def select_first_strategy():
    return select_first_strategy_()


@pytest.fixture
def max_first_model():
    return max_first_model_()


def test_get_next_move(select_first_strategy, max_first_model):
    player = AlphaPlayer(select_first_strategy, max_first_model)
    root = DummyState()
    action = player.get_next_move(root)

    assert action == root.get_possible_moves()[0]
