import pytest

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.player.pure_mcts_player.tree import *
from alpha_viergewinnt.player.pure_mcts_player.tree_search import *


class DummyState(object):
    def __init__(self):
        self.active_player = None
        self.step = 0
        self.winner = None
        self.draw = False

    def get_possible_moves(self):
        if self.step <= 3:
            return [0, 1, 2]
        else:
            return []

    def play_move(self, *args, **kwargs):
        self.step += 1

    def is_winner(self, player):
        return self.winner == player

    def is_draw(self):
        return self.step > 3


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
def simulator(dummy_strategy):
    return Simulator(strategy=dummy_strategy, player=Player.X)


def test_rollout_until_draw(simulator):
    initial_state = DummyState()
    final_state = simulator.rollout(initial_state)
    assert final_state.step == 4
    assert simulator.get_rollout_value(final_state) == 0


def test_calculate_rollout_value(simulator):
    initial_state = DummyState()
    initial_state.winner = Player.X
    assert simulator.get_rollout_value(initial_state) == 1


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
