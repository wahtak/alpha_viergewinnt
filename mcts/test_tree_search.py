import pytest

from .game_tree import GameTree
from .tree_search import *


class DummyState(object):
    def __init__(self):
        self.current_player = None
        self.step = 0

    def get_possible_moves(self):
        if self.step <= 3:
            return [0, 1, 2]
        else:
            return []

    def play_move(self, *args, **kwargs):
        self.step += 1


@pytest.fixture
def dummy_strategy():
    def first_choice_strategy(choices):
        choices = list(choices)
        choices.sort()
        return choices[0]

    return first_choice_strategy


def test_expand_tree(dummy_strategy):
    initial_state = DummyState()
    game_tree = GameTree(initial_state)

    new_state = expand_tree(game_tree, initial_state, dummy_strategy)
    assert len(game_tree.get_successors(initial_state).keys()) == 1
    assert new_state.step == 1

    expand_tree(game_tree, initial_state, dummy_strategy)
    expand_tree(game_tree, initial_state, dummy_strategy)
    assert len(game_tree.get_successors(initial_state).keys()) == 3

    with pytest.raises(NoUnexploredMovesException):
        expand_tree(game_tree, initial_state, dummy_strategy)


def test_select_leaf(dummy_strategy):
    initial_state = DummyState()
    game_tree = GameTree(initial_state)

    expand_tree(game_tree, initial_state, dummy_strategy)
    expand_tree(game_tree, initial_state, dummy_strategy)
    assert select_leaf(game_tree, initial_state, dummy_strategy).step == 0

    expand_tree(game_tree, initial_state, dummy_strategy)
    assert select_leaf(game_tree, initial_state, dummy_strategy).step == 1


@pytest.fixture
def dummy_end_condition():
    def end_condition(state):
        return len(state.get_possible_moves()) == 0

    return end_condition


def test_simulate_until_end(dummy_end_condition, dummy_strategy):
    initial_state = DummyState()

    assert simulate_until_end(initial_state, dummy_end_condition, dummy_strategy).step == 4
