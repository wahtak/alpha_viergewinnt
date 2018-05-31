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
    mcts = Mcts(graph, GameStatePath, selection_strategy=select_first_strategy, evaluation_model=max_first_model)

    return root, graph, mcts


def test_expand(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts._expand(root)
    actions = graph.get_actions(root)
    assert len(actions) == 3
    assert set(actions) == {0, 1, 2}
    assert all([graph.get_action_attributes(root, action).action_value == 0 for action in actions])
    assert all([graph.get_action_attributes(root, action).visit_count == 0 for action in actions])
    assert all([graph.get_successor(root, action).step == 1 for action in actions])
    assert all([len(graph.get_successor(root, action).played_moves) == 1 for action in actions])
    assert all([action == graph.get_successor(root, action).played_moves[0] for action in actions])

    with pytest.raises(AlreadyExpandedException):
        mcts._expand(root)


def test_select(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts._expand(root)
    path = mcts._select(root)
    assert len(path) == 2
    assert path.leaf.step == 1
    selected_action = path.get_action(root)
    assert graph.get_action_attributes(root, selected_action).visit_count == 1

    mcts._expand(path.leaf)
    path = mcts._select(root)
    assert len(path) == 3
    assert path.leaf.step == 2
    selected_action1 = path.get_action(root)
    leaf_predecessor = path.get_predecessor(path.leaf)
    selected_action2 = path.get_action(leaf_predecessor)

    assert graph.get_action_attributes(root, selected_action1).visit_count == 2
    assert graph.get_action_attributes(leaf_predecessor, selected_action2).visit_count == 1


def test_evaluate(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts._expand(root)
    mcts._evaluate(root)
    actions = graph.get_actions(root)
    assert graph.get_action_attributes(root, actions[0]).prior_probability == 1
    assert all([graph.get_action_attributes(root, action).prior_probability == 0 for action in actions[1:]])
    assert graph.get_state_attributes(root).state_value == 1


def test_backup(empty_dummy_state_mcts):
    graph = GameStateGraph(0)

    # common node
    graph.add_successor(1, source=0, action=10)
    graph.get_action_attributes(source=0, action=10).visit_count = 1
    graph.get_action_attributes(source=0, action=10).action_value = 1.0
    graph.get_state_attributes(state=1).state_value = 1.0

    # once visited leaf node
    graph.add_successor(2, source=1, action=20)
    graph.get_action_attributes(source=1, action=20).visit_count = 1
    graph.get_action_attributes(source=1, action=20).action_value = 0.5
    graph.get_state_attributes(state=2).state_value = 0.5

    # unvisited leaf node
    graph.add_successor(3, source=1, action=30)
    graph.get_action_attributes(source=1, action=30).visit_count = 0

    # simulate selection, evaluation of state 3
    path = GameStatePath(0)
    path.add_successor(1, action=10)
    graph.get_action_attributes(source=0, action=10).visit_count += 1
    path.add_successor(3, action=30)
    graph.get_action_attributes(source=1, action=30).visit_count += 1
    graph.get_state_attributes(state=3).state_value = 2.0

    mcts = Mcts(graph, GameStatePath, selection_strategy=None, evaluation_model=None)
    mcts._backup(path)

    assert graph.get_action_attributes(source=1, action=30).action_value == pytest.approx(2.0)
    assert graph.get_action_attributes(source=1, action=20).action_value == pytest.approx(0.5)
    assert graph.get_action_attributes(source=0, action=10).action_value == pytest.approx(1.25)


def test_simulate_step(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts.simulate_step(source=root)
    mcts.simulate_step(source=root)

    assert graph.get_action_attributes(source=root, action=0).visit_count == 1
    assert graph.get_action_attributes(source=root, action=1).visit_count == 0
    assert graph.get_action_attributes(source=root, action=2).visit_count == 0
    second_expanded = graph.get_successor(root, action=0)
    assert graph.get_action_attributes(source=second_expanded, action=0).visit_count == 0
    assert graph.get_action_attributes(source=second_expanded, action=1).visit_count == 0
    assert graph.get_action_attributes(source=second_expanded, action=2).visit_count == 0
    assert graph.get_state_attributes(state=second_expanded).state_value == 1
