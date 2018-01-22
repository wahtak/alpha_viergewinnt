import pytest
import matplotlib

from .graph import *


@pytest.fixture
def mcts_graph():
    return MCTSGraph(0)


def test_successor(mcts_graph):
    mcts_graph.add_successor(state=0, move=1, new_state=1)
    mcts_graph.add_successor(state=0, move=5, new_state=5)
    mcts_graph.add_successor(state=1, move=2, new_state=3)
    assert set(mcts_graph.edges()) == {(0, 1), (0, 5), (1, 3)}

    assert mcts_graph.get_successors(state=0) == {1: 1, 5: 5}
    assert mcts_graph.get_successors(state=1) == {2: 3}


def test_draw(mcts_graph):
    # use backend which does not require a display for CI
    matplotlib.use('Agg')
    mcts_graph.add_successor(state=0, move=1, new_state=1)
    mcts_graph.draw()