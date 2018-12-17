import pytest

import matplotlib

from alpha_viergewinnt.player.alpha_player.graph import *


@pytest.fixture
def graph():
    graph = GameStateGraph(root=0)
    return graph


def test_add_defaults(graph):
    graph.add_successor(20, source=0, action=10)

    assert graph.get_successor(source=0, action=10) == 20
    assert graph.get_attributes(state=0) is None


def test_add_existing_action(graph):
    graph.add_successor(1, source=0, action=10)

    with pytest.raises(ActionAlreadyExistsException):
        graph.add_successor(5, source=0, action=10)


def test_set_get_attributes(graph):
    graph.set_attributes(attributes=[1, 2, 3], state=0)

    assert graph.get_attributes(state=0) == [1, 2, 3]


def test_actions_successor_and_predecessors(graph):
    graph.add_successor(1, source=0, action=10)
    graph.add_successor(5, source=0, action=50)
    graph.add_successor(3, source=1, action=30)

    assert set(graph.get_actions(source=0)) == {10, 50}
    assert graph.get_successor(source=0, action=10) == 1
    assert graph.get_predecessors(state=1) == {0}
    assert graph.get_successor(source=0, action=50) == 5
    assert graph.get_predecessors(state=5) == {0}
    assert set(graph.get_actions(source=1)) == {30}
    assert graph.get_successor(source=1, action=30) == 3
    assert graph.get_predecessors(state=3) == {1}


def test_has_successors(graph):
    assert graph.has_successors(0) is False

    graph.add_successor(1, source=0, action=10)
    assert graph.has_successors(0) is True


def test_reset_root(graph):
    graph.add_successor(1, source=0, action=10)
    graph.add_successor(5, source=0, action=50)
    graph.add_successor(3, source=1, action=30)
    graph.add_successor(4, source=3, action=30)
    graph.set_attributes(400, state=4)
    graph.reset_root(root=1)

    assert 0 not in graph.states
    assert 0 not in sum(graph.edges, tuple())
    assert 5 not in graph.states
    assert 5 not in sum(graph.edges, tuple())
    assert 1 in graph.states
    assert graph.get_attributes(1) is None
    assert 3 in graph.states
    assert 4 in graph.states
    assert graph.get_attributes(4) is not None


def test_get_mean_node_depth(graph):
    # root at depth 0
    graph.add_successor(1, source=0, action=10)  # depth 1
    graph.add_successor(5, source=0, action=50)  # depth 1
    graph.add_successor(2, source=1, action=20)  # depth 2
    graph.add_successor(3, source=2, action=20)  # depth 3

    assert graph.get_mean_node_depth() == pytest.approx((0 + 1 + 1 + 2 + 3) / 5)


def test_draw(graph):
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    graph.add_successor(1, source=0, action=10)
    graph.draw()


@pytest.fixture
def graph_with_hashable_root():
    graph = GameStateGraph(root=HashableState(0))
    return graph


class HashableState(object):
    def __init__(self, initial_value):
        self.value = initial_value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)


def test_hash_equality_is_identity(graph_with_hashable_root):
    graph = graph_with_hashable_root
    graph.add_successor(HashableState(1), source=HashableState(0), action=1)
    graph.add_successor(HashableState(2), source=HashableState(0), action=2)
    graph.add_successor(HashableState(3), source=HashableState(1), action=3)
    graph.add_successor(HashableState(3), source=HashableState(2), action=3)

    assert len(graph.states) == 4
    assert graph.get_predecessors(state=HashableState(3)) == {HashableState(1), HashableState(2)}
