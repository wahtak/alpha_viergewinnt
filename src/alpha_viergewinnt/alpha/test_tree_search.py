import pytest

from .tree import *
from .tree_search import *


class DummyState(object):
    def __init__(self):
        self.current_player = None
        self.played_moves = []
        self.step = 0

    def __hash__(self):
        return hash(tuple(self.played_moves))

    def __eq__(self, other):
        return self.played_moves == other.played_moves

    def get_possible_moves(self):
        if self.step <= 3:
            return [0, 1, 2]
        else:
            return []

    def play_move(self, player, move):
        self.step += 1
        self.played_moves.append(move)


@pytest.fixture
def select_first_strategy():
    def select_first(actions, attributes):
        actions = list(actions)
        actions.sort()
        return actions[0]

    return select_first


@pytest.fixture
def max_first_model():
    def evaluate(actions, state):
        max_first_prior_probabilities = [1] + ([0] * (len(actions) - 1))
        dummy_state_value = 1
        return max_first_prior_probabilities, dummy_state_value

    return evaluate


@pytest.fixture
def empty_dummy_state_tree_search(select_first_strategy, max_first_model):
    root = DummyState()
    tree = Tree(root)
    tree_search = TreeSearch(tree, selection_strategy=select_first_strategy, evaluation_model=max_first_model)

    return root, tree, tree_search


def test_expand(empty_dummy_state_tree_search):
    root, tree, tree_search = empty_dummy_state_tree_search

    tree_search.expand(root)
    actions = tree.get_actions(root)
    assert len(actions) == 3
    assert set(actions) == {0, 1, 2}
    assert all([tree.get_attributes(source=root, action=action).action_value is None for action in actions])
    assert all([tree.get_attributes(source=root, action=action).visit_count == 0 for action in actions])
    assert all([tree.get_successor(source=root, action=action).step == 1 for action in actions])
    assert all([len(tree.get_successor(source=root, action=action).played_moves) == 1 for action in actions])
    assert all([action == tree.get_successor(source=root, action=action).played_moves[0] for action in actions])

    with pytest.raises(AlreadyExpandedException):
        tree_search.expand(root)


def test_select_leaf(empty_dummy_state_tree_search):
    root, tree, tree_search = empty_dummy_state_tree_search

    tree_search.expand(root)
    selected_leaf1 = tree_search.select_leaf(root)
    # assume selection strategy always picks first move
    selected_leaf1_action = root.get_possible_moves()[0]
    assert selected_leaf1.step == 1
    assert tree.get_attributes(source=root, action=selected_leaf1_action).visit_count == 1

    tree_search.expand(selected_leaf1)
    selected_leaf2 = tree_search.select_leaf(root)
    # assume selection strategy always picks first move
    selected_leaf1_action = root.get_possible_moves()[0]
    selected_leaf2_action = selected_leaf1.get_possible_moves()[0]

    assert selected_leaf2.step == 2
    assert tree.get_attributes(source=root, action=selected_leaf1_action).visit_count == 2
    assert tree.get_attributes(source=selected_leaf1, action=selected_leaf2_action).visit_count == 1


def test_evaluate(empty_dummy_state_tree_search):
    root, tree, tree_search = empty_dummy_state_tree_search

    tree_search.expand(root)
    tree_search.evaluate(root)
    actions = tree.get_actions(root)
    assert tree.get_attributes(source=root, action=actions[0]).prior_probability == 1
    assert all([tree.get_attributes(source=root, action=action).prior_probability == 0 for action in actions[1:]])
    assert tree.get_attributes(source=root).state_value == 1
