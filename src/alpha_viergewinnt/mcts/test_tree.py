import pytest
import matplotlib

from .tree import *


@pytest.fixture
def tree():
    return Tree(0)


def test_successor_and_ancestors(tree):
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


def test_draw(tree):
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    tree.add_successor(source=0, transition=1, successor=1)
    tree.draw()


def test_attributes(tree):
    assert tree.attributes[0].visit_count == 0
    assert tree.attributes[0].weight == 0
