import pytest
import matplotlib

from .game_tree import *


@pytest.fixture
def game_tree():
    return GameTree(0)


def test_successor_and_ancestors(game_tree):
    game_tree.add_successor(state=0, move=1, new_state=1)
    game_tree.add_successor(state=0, move=5, new_state=5)
    game_tree.add_successor(state=1, move=2, new_state=3)
    assert set(game_tree.edges()) == {(0, 1), (0, 5), (1, 3)}

    assert game_tree.get_successors(state=0) == {1: 1, 5: 5}
    assert game_tree.get_successors(state=1) == {2: 3}
    assert game_tree.get_ancestors(state=3) == {0, 1}


def test_draw(game_tree):
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    game_tree.add_successor(state=0, move=1, new_state=1)
    game_tree.draw()


def test_attributes(game_tree):
    assert game_tree.attributes[0].visit_count == 0
    assert game_tree.attributes[0].weight == 0
