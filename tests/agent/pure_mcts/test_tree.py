import matplotlib

from alpha_viergewinnt.agent.pure_mcts.tree import *


def test_successor_and_ancestors():
    tree = Tree(0)
    tree.add_successor(source=0, transition=1, successor=1)
    tree.add_successor(source=0, transition=5, successor=5)
    tree.add_successor(source=1, transition=2, successor=3)
    assert set(tree.edges()) == {(0, 1), (0, 5), (1, 3)}

    assert tree.get_transitions(source=0) == {1, 5}
    assert tree.get_successor(source=0, transition=1) == 1
    assert tree.get_successor(source=0, transition=5) == 5
    assert tree.get_transitions(source=1) == {2}
    assert tree.get_successor(source=1, transition=2) == 3
    assert tree.get_path_to_root(source=3) == {0, 1, 3}


def test_draw():
    tree = Tree(0)
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    tree.add_successor(source=0, transition=1, successor=1)
    tree.draw()


def test_attributes():
    tree = Tree(0)
    tree.add_successor(source=0, transition=10, successor=1)
    tree.add_successor(source=1, transition=20, successor=2)
    tree.add_successor(source=1, transition=30, successor=3)

    assert tree.attributes[0].visit_count == 0
    assert tree.attributes[0].weight == 0
    tree.attributes[2].weight = 1
    assert tree.get_transition_to_max_weight(1) == 20


class HashableState(object):
    def __init__(self, initial_value):
        self.value = initial_value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)


def test_hash_equality_is_identity():
    tree = Tree(HashableState(0))
    tree.add_successor(source=HashableState(0), transition=1, successor=HashableState(1))
    tree.add_successor(source=HashableState(1), transition=3, successor=HashableState(3))
    tree.add_successor(source=HashableState(0), transition=2, successor=HashableState(2))
    tree.add_successor(source=HashableState(2), transition=3, successor=HashableState(3))
    assert len(tree.nodes()) == 4
    assert len(tree.attributes) == 4

    assert tree.get_path_to_root(source=HashableState(3)) == {
        HashableState(0), HashableState(1), HashableState(2),
        HashableState(3)
    }
