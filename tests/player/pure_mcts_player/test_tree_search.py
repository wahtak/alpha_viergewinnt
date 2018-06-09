import pytest

from alpha_viergewinnt.player.pure_mcts_player.tree import *
from alpha_viergewinnt.player.pure_mcts_player.tree_search import *


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

    def check(self, condition):
        return condition.check(self)


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
def no_possible_moves_condition():
    class NoPossibleMovesCondition(object):
        def check(self, state):
            return len(state.get_possible_moves()) == 0

    return NoPossibleMovesCondition()


@pytest.fixture
def false_condition():
    class FalseCondition(object):
        def check(self, _):
            return False

    return FalseCondition()


@pytest.fixture
def true_condition():
    class TrueCondition(object):
        def check(self, _):
            return True

    return TrueCondition()


@pytest.fixture
def rollout_until_draw_simulator(dummy_strategy, no_possible_moves_condition, false_condition):
    return Simulator(
        strategy=dummy_strategy,
        win_condition=false_condition,
        loss_condition=false_condition,
        draw_condition=no_possible_moves_condition)


@pytest.fixture
def immediate_win_simulator(dummy_strategy, no_possible_moves_condition, false_condition, true_condition):
    return Simulator(
        strategy=dummy_strategy,
        win_condition=true_condition,
        loss_condition=false_condition,
        draw_condition=no_possible_moves_condition)


def test_rollout_until_draw(rollout_until_draw_simulator):
    initial_state = DummyState()
    simulator = rollout_until_draw_simulator
    final_state = simulator.rollout(initial_state)
    assert final_state.step == 4
    assert simulator.get_rollout_value(final_state) == 0


def test_calculate_rollout_value(immediate_win_simulator):
    initial_state = DummyState()
    simulator = immediate_win_simulator
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
