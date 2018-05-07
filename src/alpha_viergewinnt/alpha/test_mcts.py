import pytest

from .graph import *
from .mcts import *


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
def empty_dummy_state_mcts(select_first_strategy, max_first_model):
    root = DummyState()
    graph = GameStateGraph(root)
    mcts = Mcts(graph, selection_strategy=select_first_strategy, evaluation_model=max_first_model)

    return root, graph, mcts


def test_expand(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts.expand(root)
    actions = graph.get_actions(root)
    assert len(actions) == 3
    assert set(actions) == {0, 1, 2}
    assert all([graph.get_transition_attributes(root, action).action_value is None for action in actions])
    assert all([graph.get_transition_attributes(root, action).visit_count == 0 for action in actions])
    assert all([graph.get_successor(root, action).step == 1 for action in actions])
    assert all([len(graph.get_successor(root, action).played_moves) == 1 for action in actions])
    assert all([action == graph.get_successor(root, action).played_moves[0] for action in actions])

    with pytest.raises(AlreadyExpandedException):
        mcts.expand(root)


def test_select_leaf(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts.expand(root)
    selected_leaf1 = mcts.select_leaf(root)
    # assume selection strategy always picks first move
    selected_leaf1_action = root.get_possible_moves()[0]
    assert selected_leaf1.step == 1
    assert graph.get_transition_attributes(root, selected_leaf1_action).visit_count == 1

    mcts.expand(selected_leaf1)
    selected_leaf2 = mcts.select_leaf(root)
    # assume selection strategy always picks first move
    selected_leaf1_action = root.get_possible_moves()[0]
    selected_leaf2_action = selected_leaf1.get_possible_moves()[0]

    assert selected_leaf2.step == 2
    assert graph.get_transition_attributes(root, selected_leaf1_action).visit_count == 2
    assert graph.get_transition_attributes(selected_leaf1, selected_leaf2_action).visit_count == 1


def test_evaluate(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts.expand(root)
    mcts.evaluate(root)
    actions = graph.get_actions(root)
    assert graph.get_transition_attributes(root, actions[0]).prior_probability == 1
    assert all([graph.get_transition_attributes(root, action).prior_probability == 0 for action in actions[1:]])
    assert graph.get_state_attributes(root).state_value == 1


@pytest.mark.skip()
def test_backup(empty_dummy_state_mcts):
    graph = GameStateGraph(0)
    graph.add_successor(1, source=0, action=10)
    graph.get_transition_attributes(source=0, action=10).visit_count = 1
    graph.get_transition_attributes(source=0, action=10).action_value = 1.0
    graph.get_state_attributes(state=1).state_value = 1.0

    # once visited leaf node
    graph.add_successor(2, source=1, action=20)
    graph.get_transition_attributes(source=1, action=20).visit_count = 1
    graph.get_transition_attributes(source=1, action=20).action_value = 0.5
    graph.get_state_attributes(state=2).state_value = 0.5

    # unvisited leaf node
    graph.add_successor(3, source=1, action=30)
    graph.get_transition_attributes(source=1, action=30).visit_count = 0

    # simulate selection, evaluation of state 3
    graph.get_transition_attributes(source=0, action=10).visit_count += 1
    graph.get_transition_attributes(source=1, action=30).visit_count += 1
    graph.get_state_attributes(state=3).state_value = 2.0

    mcts = Mcts(graph, selection_strategy=None, evaluation_model=None)
    mcts.backup(3)

    assert graph.get_transition_attributes(source=1, action=30).action_value == pytest.approx(2.0)
    assert graph.get_transition_attributes(source=1, action=20).action_value == pytest.approx(0.5)
    assert graph.get_transition_attributes(source=0, action=10).action_value == pytest.approx(1.25)
