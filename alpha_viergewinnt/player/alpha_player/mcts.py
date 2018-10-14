from copy import deepcopy

import numpy as np

from .attributes import Attributes


class Mcts(object):
    def __init__(self, graph, evaluation_model):
        self.graph = graph
        self.evaluation_model = evaluation_model

    def simulate_step(self, source):
        selected_path = self._select_path(source)
        self._expand(selected_path.leaf)
        self._backup(selected_path)

    def _select_path(self, source):
        """
        Select a path from source to a leaf state in graph according to selection strategy.
        """
        path = self.graph.create_path(source)
        state = source
        while not self._is_leaf(state):
            selected_action = self._select_action(state)
            successor = self.graph.get_successor(state, selected_action)
            path.add_successor(successor, selected_action)
            state = successor
        return path

    def _is_leaf(self, state):
        return not self.graph.has_successors(state)

    def _select_action(self, state):
        potential_value = self._get_potential_value(state)
        masked_potential_value = self._mask_invalid_actions(potential_value, state, fill_value=np.NINF)
        return np.argmax(masked_potential_value)

    def _get_potential_value(self, state):
        exploration_factor = 1
        attributes = self.graph.get_attributes(state)
        upper_confidence_bound = exploration_factor * attributes.prior_distribution * np.sqrt(
            np.sum(attributes.visit_count)) / (1 + attributes.visit_count)
        return attributes.action_value + upper_confidence_bound

    def _mask_invalid_actions(self, values, state, fill_value):
        actions = self.graph.get_actions(state)
        masked_values = np.full_like(values, fill_value)
        masked_values[actions] = values[actions]
        return masked_values

    def _expand(self, leaf):
        """
        Evaluate the state value and prior probabilities for all actions with the evaluation model
        and add resulting actions and states.
        """
        assert self._is_leaf(leaf)

        actions = leaf.get_possible_moves()
        prior_distribution, state_value, game_finished = self.evaluation_model(actions, leaf)

        attributes = Attributes(state_value, prior_distribution)
        self.graph.set_attributes(attributes, state=leaf)

        if not game_finished:
            for action in actions:
                successor = deepcopy(leaf)
                successor.play_move(player=leaf.active_player, move=action)
                self.graph.add_successor(successor, source=leaf, action=action)

    def _backup(self, path):
        """
        Backpropagate the state value up a (previously selected) path, by updating the action values and visit count
        """
        path_state = path.leaf
        leaf_value = self.graph.get_attributes(path.leaf).state_value
        while path_state is not path.root:
            path_state = path.get_predecessor(path_state)
            path_action = path.get_action(path_state)
            self._update_attributes(path_state, path_action, action_value_update=leaf_value)

    def _update_attributes(self, state, action, action_value_update):
        visit_count = self.graph.get_attributes(state).visit_count
        action_value = self.graph.get_attributes(state).action_value

        total_action_value = action_value[action] * visit_count[action]
        total_action_value += action_value_update
        visit_count[action] += 1
        action_value[action] = total_action_value / visit_count[action]

    def get_prior_distribution(self, state):
        self.graph.get_attributes(state).prior_distribution

    def get_search_distribution(self, state, exploration_factor):
        visit_count = self.graph.get_attributes(state).visit_count
        masked_visit_count = self._mask_invalid_actions(visit_count, state, fill_value=0)
        assert np.sum(masked_visit_count) != 0
        # smaller exploration factor leads to numerical wierdness
        assert exploration_factor >= 0.01
        likelihoods = np.power(masked_visit_count, 1 / exploration_factor)
        probabilities = likelihoods / sum(likelihoods)
        return probabilities
