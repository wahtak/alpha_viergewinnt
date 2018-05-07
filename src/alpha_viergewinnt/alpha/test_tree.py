import pytest

import matplotlib

from .tree import *


def test_add_defaults():
    tree = Tree(0)
    tree.add_successor(20, source=0, action=10)

    assert tree.get_successor(source=0, action=10) == 20
    assert tree.get_transition_attributes(source=0, action=10).action_value is None
    assert tree.get_transition_attributes(source=0, action=10).prior_probability is None
    assert tree.get_transition_attributes(source=0, action=10).visit_count == 0


def test_add_existing_action():
    tree = Tree(0)
    tree.add_successor(1, source=0, action=10)

    with pytest.raises(ActionAlreadyExistsException):
        tree.add_successor(5, source=0, action=10)


def test_attributes():
    tree = Tree(0)
    tree.add_successor(1, source=0, action=10)
    tree.get_transition_attributes(source=0, action=10).action_value = 5
    tree.get_transition_attributes(source=0, action=10).prior_probability = 1
    tree.get_transition_attributes(source=0, action=10).visit_count = 2
    tree.get_state_attributes(state=1).state_value = 1

    assert tree.get_transition_attributes(source=0, action=10).action_value == 5
    assert tree.get_transition_attributes(source=0, action=10).prior_probability == 1
    assert tree.get_transition_attributes(source=0, action=10).visit_count == 2
    assert tree.get_state_attributes(state=1).state_value == 1


def test_actions_successor_and_path_to_root():
    tree = Tree(0)
    tree.add_successor(1, source=0, action=10)
    tree.add_successor(5, source=0, action=50)
    tree.add_successor(3, source=1, action=30)

    assert set(tree.get_actions(source=0)) == {10, 50}
    assert tree.get_successor(source=0, action=10) == 1
    assert tree.get_successor(source=0, action=50) == 5
    assert set(tree.get_actions(source=1)) == {30}
    assert tree.get_successor(source=1, action=30) == 3
    assert tree.get_path_to_root(source=3) == {0, 1, 3}


def test_has_successors():
    tree = Tree(0)

    assert tree.has_successors(0) is False

    tree.add_successor(1, source=0, action=10)
    assert tree.has_successors(0) is True


def test_draw():
    tree = Tree(0)
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    tree.add_successor(1, source=0, action=10)
    tree.draw()


class HashableState(object):
    def __init__(self, initial_value):
        self.value = initial_value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)


def test_hash_equality_is_identity():
    tree = Tree(HashableState(0))
    tree.add_successor(HashableState(1), source=HashableState(0), action=1)
    tree.add_successor(HashableState(2), source=HashableState(0), action=2)
    tree.add_successor(HashableState(3), source=HashableState(1), action=3)
    tree.add_successor(HashableState(3), source=HashableState(2), action=3)

    assert len(tree.states) == 4
    assert tree.get_path_to_root(source=HashableState(3)) == {
        HashableState(0), HashableState(1), HashableState(2),
        HashableState(3)
    }
