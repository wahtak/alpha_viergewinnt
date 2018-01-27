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


@pytest.fixture
def filled_simple_state_game_tree():
    game_tree = GameTree((0, ))
    game_tree.add_successor(state=(0, ), move=0, new_state=(0, 0))
    game_tree.add_successor(state=(0, ), move=1, new_state=(0, 1))
    game_tree.add_successor(state=(0, 1), move=0, new_state=(0, 1, 0))
    game_tree.add_successor(state=(0, 1, 0), move=0, new_state=(0, 1, 0, 0))
    return game_tree


@pytest.fixture
def simple_state_leafs():
    return (0, 1, 0, 0), (0, 0)


def test_backpropagate(filled_simple_state_game_tree, simple_state_leafs):
    game_tree = filled_simple_state_game_tree
    update_leaf, other_leaf = simple_state_leafs

    backpropagate(game_tree, update_leaf, delta_weight=-2)
    backpropagate(game_tree, update_leaf, delta_weight=1)

    assert game_tree.attributes[update_leaf].visit_count == 2
    assert game_tree.attributes[update_leaf].weight == -1

    for state in game_tree.get_ancestors(update_leaf):
        assert game_tree.attributes[state].visit_count == 2
        assert game_tree.attributes[state].weight == -1

    assert game_tree.attributes[other_leaf].visit_count == 0
    assert game_tree.attributes[other_leaf].weight == 0
