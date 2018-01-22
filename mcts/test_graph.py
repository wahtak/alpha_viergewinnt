import pytest
import matplotlib

from .graph import *


@pytest.fixture
def game_graph():
    return GameGraph(0)


def test_successor(game_graph):
    game_graph.add_successor(state=0, move=1, new_state=1)
    game_graph.add_successor(state=0, move=5, new_state=5)
    game_graph.add_successor(state=1, move=2, new_state=3)
    assert set(game_graph.edges()) == {(0, 1), (0, 5), (1, 3)}

    assert game_graph.get_successors(state=0) == {1: 1, 5: 5}
    assert game_graph.get_successors(state=1) == {2: 3}


def test_draw(game_graph):
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    game_graph.add_successor(state=0, move=1, new_state=1)
    game_graph.draw()