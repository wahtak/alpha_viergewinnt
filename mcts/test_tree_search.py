import pytest

from .tree import *
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


@pytest.fixture
def empty_dummy_state_tree_search(dummy_strategy):
    root = DummyState()
    tree = Tree(root)
    tree_search = TreeSearch(tree, selection_strategy=dummy_strategy, expansion_strategy=dummy_strategy)

    return root, tree, tree_search


def test_expand(empty_dummy_state_tree_search):
    root, tree, tree_search = empty_dummy_state_tree_search

    succesor = tree_search.expand(root)
    assert len(tree.get_transitions(root)) == 1
    assert succesor.step == 1

    tree_search.expand(root)
    tree_search.expand(root)
    assert len(tree.get_transitions(root)) == 3

    with pytest.raises(NoUnexploredMovesException):
        tree_search.expand(root)


def test_select_leaf(empty_dummy_state_tree_search):
    root, tree, tree_search = empty_dummy_state_tree_search

    tree_search.expand(root)
    tree_search.expand(root)
    assert tree_search.select_leaf(root).step == 0

    tree_search.expand(root)
    assert tree_search.select_leaf(root).step == 1


@pytest.fixture
def dummy_end_condition():
    def end_condition(state):
        return len(state.get_possible_moves()) == 0

    return end_condition


@pytest.fixture
def simulator(dummy_strategy, dummy_end_condition):
    return Simulator(strategy=dummy_strategy, end_condition=dummy_end_condition)


def test_simulate_until_end(simulator):
    initial_state = DummyState()
    assert simulator.simulate_until_end(initial_state).step == 4


@pytest.fixture
def filled_simple_state_tree_search(dummy_strategy):
    tree = Tree((0, ))
    tree.add_successor(source=(0, ), transition=0, successor=(0, 0))
    tree.add_successor(source=(0, ), transition=1, successor=(0, 1))
    tree.add_successor(source=(0, 1), transition=0, successor=(0, 1, 0))
    tree.add_successor(source=(0, 1, 0), transition=0, successor=(0, 1, 0, 0))
    tree_search = TreeSearch(tree, selection_strategy=dummy_strategy, expansion_strategy=dummy_strategy)
    return tree, tree_search


@pytest.fixture
def simple_state_leafs():
    return (0, 1, 0, 0), (0, 0)


def test_backpropagate(filled_simple_state_tree_search, simple_state_leafs):
    tree, tree_search = filled_simple_state_tree_search
    update_leaf, other_leaf = simple_state_leafs

    tree_search.backpropagate(update_leaf, delta_weight=-2)
    tree_search.backpropagate(update_leaf, delta_weight=1)

    assert tree.attributes[update_leaf].visit_count == 2
    assert tree.attributes[update_leaf].weight == -1

    for state in tree.get_path_to_root(update_leaf):
        assert tree.attributes[state].visit_count == 2
        assert tree.attributes[state].weight == -1

    assert tree.attributes[other_leaf].visit_count == 0
    assert tree.attributes[other_leaf].weight == 0
