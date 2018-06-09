from random import Random
from copy import deepcopy

from .tree import Tree
from .tree_search import TreeSearch, Simulator


def create_random_choice_strategy(random=Random()):
    def random_choice_strategy(moves):
        return random.choice(list(moves))

    return random_choice_strategy


class PureMctsPlayer(object):
    def __init__(self, win_condition, loss_condition, draw_condition, selection_strategy, expansion_strategy,
                 simulation_strategy, iterations, rollouts, **kwargs):

        self.win_condition = win_condition
        self.loss_condition = loss_condition
        self.draw_condition = draw_condition
        self.selection_strategy = selection_strategy
        self.expansion_strategy = expansion_strategy
        self.simulation_strategy = simulation_strategy
        self.iterations = iterations
        self.rollouts = rollouts

        self.simulator = Simulator(
            strategy=self.simulation_strategy,
            win_condition=self.win_condition,
            loss_condition=self.loss_condition,
            draw_condition=self.draw_condition)
        self._last_tree = None

    def get_next_move(self, state):
        initial_state = state
        tree = Tree(initial_state)
        tree_search = TreeSearch(
            tree, selection_strategy=self.selection_strategy, expansion_strategy=self.expansion_strategy)

        self._explore_tree_and_update_weights(tree_search, initial_state)
        self._last_tree = tree
        return tree.get_transition_to_max_weight(initial_state)

    def _explore_tree_and_update_weights(self, tree_search, initial_state):
        for _ in range(self.iterations):
            leaf_state = tree_search.select_leaf(initial_state)
            if not self._is_final_state(leaf_state):
                expanded_state = tree_search.expand(leaf_state)
            else:
                expanded_state = leaf_state
            state_utility = self._get_state_utility(expanded_state)
            tree_search.backpropagate(expanded_state, state_utility)

    def _is_final_state(self, state):
        return state.check(self.win_condition) or state.check(self.loss_condition) or state.check(self.draw_condition)

    def _get_state_utility(self, state):
        rollout_value_sum = 0
        for _ in range(self.rollouts):
            initial_state = deepcopy(state)
            final_state = self.simulator.rollout(initial_state)
            rollout_value_sum += self.simulator.get_rollout_value(final_state)
        return rollout_value_sum / self.rollouts

    def draw_last_tree(self):
        self._last_tree.draw()