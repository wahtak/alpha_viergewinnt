from copy import deepcopy

import numpy as np

from .attributes import Attributes


class AlreadyExpandedException(Exception):
    pass


class Mcts(object):
    def __init__(self, graph, path_factory, evaluation_model):
        self.graph = graph
        self.path_factory = path_factory
        self.evaluation_model = evaluation_model

    def simulate_step(self, source):
        selected_path = self._select_path(source)
        self._expand(selected_path.leaf)
        self._backup(selected_path)

    def _select_path(self, source):
        """
        Select a path from source to a leaf state in graph according to selection strategy.
        """
        path = self.path_factory(source)
        state = source
        while self.graph.has_successors(state):
            selected_action = self._select_action(state)
            self.graph.get_attributes(state).visit_counts[selected_action] += 1
            successor = self.graph.get_successor(state, selected_action)
            path.add_successor(successor, selected_action)
            state = successor
        return path

    def _select_action(self, state):
        potential_values = self._get_potential_values(state)
        masked_potential_values = self._mask_invalid_actions(potential_values, state, fill_value=np.NINF)
        return np.argmax(masked_potential_values)

    def _get_potential_values(self, state):
        exploration_factor = 1
        attributes = self.graph.get_attributes(state)
        upper_confidence_bound = exploration_factor * attributes.prior_probabilities * np.sqrt(
            np.sum(attributes.visit_counts)) / (1 + attributes.visit_counts)
        return attributes.action_values + upper_confidence_bound

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
        if self.graph.has_successors(leaf):
            raise AlreadyExpandedException()

        actions = leaf.get_possible_moves()
        prior_probabilities, state_value, game_finished = self.evaluation_model(actions, leaf)

        attributes = Attributes(state_value, prior_probabilities)
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
        subtree_value = self.graph.get_attributes(path.leaf).state_value
        while path_state is not path.root:
            path_state = path.get_predecessor(path_state)
            path_action = path.get_action(path_state)
            action_values = self.graph.get_attributes(path_state).action_values
            action_values[path_action] = subtree_value
            subtree_value = np.mean(action_values)

    def get_search_probabilities(self, state, exploration_factor):
        visit_counts = self.graph.get_attributes(state).visit_counts
        masked_visit_counts = self._mask_invalid_actions(visit_counts, state, fill_value=0)
        assert np.sum(masked_visit_counts) != 0
        if exploration_factor == 0:
            probabilities = np.zeros_like(masked_visit_counts)
            probabilities[np.argmax(masked_visit_counts)] = 1.0
        else:
            likelihoods = np.power(masked_visit_counts, 1 / exploration_factor)
            probabilities = likelihoods / sum(likelihoods)
        return probabilities
