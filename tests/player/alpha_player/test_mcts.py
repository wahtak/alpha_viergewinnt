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
def max_first_model():
    def evaluate(actions, state):
        max_first_prior_probabilities = np.array([1] + ([0] * (len(actions) - 1)))
        dummy_state_value = 1
        game_finished = False
        return max_first_prior_probabilities, dummy_state_value, game_finished

    return evaluate


@pytest.fixture
def empty_dummy_state_mcts(max_first_model):
    root = DummyState()
    graph = GameStateGraph(root)
    mcts = Mcts(graph, evaluation_model=max_first_model)

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


def test_select_path(empty_dummy_state_mcts):
    root, graph, mcts = empty_dummy_state_mcts

    mcts._expand(root)
    path = mcts._select_path(root)
    assert len(path) == 2
    assert path.leaf.step == 1
    assert path.get_predecessor(path.leaf) == root
    assert path.get_action(root) == 0

    mcts._expand(path.leaf)
    path = mcts._select_path(root)
    assert len(path) == 3
    assert path.leaf.step == 2
    predecessor = path.get_predecessor(path.leaf)
    assert predecessor.step == 1
    assert path.get_action(predecessor) == 0
    assert path.get_predecessor(predecessor) == root


def test_select_action(empty_dummy_state_mcts):
    graph = GameStateGraph('r')

    graph.add_successor('r.0', source='r', action=0)
    graph.add_successor('r.1', source='r', action=1)
    graph.add_successor('r.3', source='r', action=3)
    attributes = Attributes(state_value=None, prior_probabilities=np.array([1, 1, 1, 0.1]))
    attributes.action_values = np.array([0.1, 0.1, 0.2, 0])
    attributes.visit_counts = np.array([10, 1, 0, 0])
    graph.set_attributes(attributes, state='r')

    mcts = Mcts(graph, evaluation_model=None)
    assert mcts._select_action(state='r') == 1


def test_backup():
    graph = GameStateGraph('r')

    common_node_state_value = 0.4
    expanded_node_state_value = -0.6
    selected_node_state_value = 1.0

    # add root node
    root_node_attributes = Attributes(state_value=0.0, prior_probabilities=[None, None])
    graph.set_attributes(root_node_attributes, state='r')

    # add common node
    common_node_attributes = Attributes(state_value=common_node_state_value, prior_probabilities=[None, None])
    graph.add_successor('r.0', source='r', action=0)
    graph.set_attributes(common_node_attributes, state='r.0')
    root_node_attributes.visit_counts[0] = 2
    root_node_attributes.action_values[0] = (common_node_state_value + expanded_node_state_value) / 2

    # add expanded leaf node
    expanded_node_attributes = Attributes(state_value=expanded_node_state_value, prior_probabilities=[None, None])
    graph.add_successor('r.0.0', source='r.0', action=0)
    graph.set_attributes(expanded_node_attributes, state='r.0.0')
    common_node_attributes.visit_counts[0] = 1
    common_node_attributes.action_values[0] = expanded_node_state_value

    # add unvisited leaf node
    graph.add_successor('r.0.1', source='r.0', action=1)

    # simulate selection of unvisited leaf node
    path = GameStatePath('r')
    path.add_successor('r.0', action=0)
    path.add_successor('r.0.1', action=1)

    # simulate expansion of selected leaf node
    selected_node_attributes = Attributes(state_value=selected_node_state_value, prior_probabilities=[None, None])
    graph.set_attributes(selected_node_attributes, state='r.0.1')

    mcts = Mcts(graph, evaluation_model=None)
    mcts._backup(path)

    # expect the attributes of an action which was not selected to be unmodified
    assert graph.get_attributes(state='r.0').action_values[0] == pytest.approx(expanded_node_state_value)
    assert graph.get_attributes(state='r.0').visit_counts[0] == 1

    # expect the action_value of an action resulting in a leaf state to equal to the state_value
    assert graph.get_attributes(state='r.0').action_values[1] == pytest.approx(selected_node_state_value)
    # expect the visit count of an expanded state to be 1
    assert graph.get_attributes(state='r.0').visit_counts[1] == 1

    # expect the action_value of an action resulting in a non-leaf state to equal to be the mean
    # of its state value and its subtree action values
    mean_action_value = (common_node_state_value + expanded_node_state_value + selected_node_state_value) / 3
    assert graph.get_attributes(state='r').action_values[0] == pytest.approx(mean_action_value)
    assert graph.get_attributes(state='r').visit_counts[0] == 3


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
