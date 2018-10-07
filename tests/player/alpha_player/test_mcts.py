import pytest

from alpha_viergewinnt.player.alpha_player.graph import *
from alpha_viergewinnt.player.alpha_player.mcts import *


class DummyState(object):
    def __init__(self):
        self.active_player = None
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
        max_first_prior_probabilities = np.array([1] + ([0] * (len(actions) - 1)))
        dummy_state_value = 1
        game_finished = False
        return max_first_prior_probabilities, dummy_state_value, game_finished

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
    attributes = graph.get_attributes(root)
    assert len(actions) == 3
    assert set(actions) == {0, 1, 2}
    assert all(attributes.action_values == 0)
    assert all(attributes.visit_counts == 0)
    assert all([graph.get_successor(root, action).step == 1 for action in actions])
    assert all([len(graph.get_successor(root, action).played_moves) == 1 for action in actions])
    assert all([action == graph.get_successor(root, action).played_moves[0] for action in actions])
    assert attributes.prior_probabilities[0] == 1
    assert all(attributes.prior_probabilities[1:] == 0)
    assert attributes.state_value == 1

    with pytest.raises(AlreadyExpandedException):
        mcts._expand(root)


def test_select(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts._expand(root)
    path = mcts._select(root)
    assert len(path) == 2
    assert path.leaf.step == 1
    selected_action = path.get_action(root)
    assert graph.get_attributes(root).visit_counts[selected_action] == 1

    mcts._expand(path.leaf)
    path = mcts._select(root)
    assert len(path) == 3
    assert path.leaf.step == 2
    selected_action1 = path.get_action(root)
    leaf_predecessor = path.get_predecessor(path.leaf)
    selected_action2 = path.get_action(leaf_predecessor)

    assert graph.get_attributes(root).visit_counts[selected_action1] == 2
    assert graph.get_attributes(leaf_predecessor).visit_counts[selected_action2] == 1


def test_backup(empty_dummy_state_mcts):
    graph = GameStateGraph(0)

    # expand root node
    attributes = Attributes(state_value=None, prior_probabilities=[None, None])
    graph.set_attributes(attributes, state=0)
    # add common node
    graph.add_successor(1, source=0, action=0)
    attributes.action_values[0] = -0.3

    # expand common node
    attributes = Attributes(state_value=None, prior_probabilities=[None, None])
    graph.set_attributes(attributes, state=1)
    # add expanded leaf node
    graph.add_successor(2, source=1, action=0)
    attributes.action_values[0] = -0.6
    # add unvisited leaf node
    graph.add_successor(3, source=1, action=1)

    # simulate selection state 3
    path = GameStatePath(0)
    path.add_successor(1, action=0)
    path.add_successor(3, action=1)

    # simulate expansion of state 3 which turn out to be a winning state
    attributes = Attributes(state_value=1.0, prior_probabilities=[None, None])
    graph.set_attributes(attributes, state=3)

    mcts = Mcts(graph, GameStatePath, selection_strategy=None, evaluation_model=None)
    mcts._backup(path)

    # expect the action_value of an action which was not selected to be unmodified
    assert graph.get_attributes(state=1).action_values[0] == pytest.approx(-0.6)

    # expect the action_value of an action resulting in a leaf state to equal to the state_value
    assert graph.get_attributes(state=1).action_values[1] == pytest.approx(1.0)

    # expect the action_value of an action resulting in a non-leaf state to equal to the mean of sub action values
    assert graph.get_attributes(state=0).action_values[0] == pytest.approx(0.2)


def test_simulate_step(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts.simulate_step(source=root)
    mcts.simulate_step(source=root)

    assert graph.get_attributes(state=root).visit_counts[0] == 1
    assert graph.get_attributes(state=root).visit_counts[1] == 0
    assert graph.get_attributes(state=root).visit_counts[2] == 0
    second_expanded = graph.get_successor(root, action=0)
    assert graph.get_attributes(state=second_expanded).visit_counts[0] == 0
    assert graph.get_attributes(state=second_expanded).visit_counts[1] == 0
    assert graph.get_attributes(state=second_expanded).visit_counts[2] == 0
    assert graph.get_attributes(state=second_expanded).state_value == 1
