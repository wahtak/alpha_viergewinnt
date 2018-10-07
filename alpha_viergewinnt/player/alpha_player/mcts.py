from copy import deepcopy

import numpy as np

from .attributes import Attributes


class AlreadyExpandedException(Exception):
    pass


class Mcts(object):
    def __init__(self, graph, path_factory, selection_strategy, evaluation_model):
        self.graph = graph
        self.path_factory = path_factory
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model

    def simulate_step(self, source):
        selected_path = self._select(source)
        self._expand(selected_path.leaf)
        self._backup(selected_path)

    def _select(self, source):
        """
        Select a path from source to a leaf state in graph according to selection strategy.
        """
        path = self.path_factory(source)
        state = source
        while self.graph.has_successors(state):
            actions = self.graph.get_actions(state)
            attributes = self.graph.get_attributes(state)
            selected_action = self.selection_strategy(actions, attributes)
            attributes.visit_counts[selected_action] += 1
            successor = self.graph.get_successor(state, selected_action)
            path.add_successor(successor, selected_action)
            state = successor
        return path

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
        Backpropagate the state value up a (previously selected) path, by updating the action values
        """
        path_state = path.leaf
        subtree_value = self.graph.get_attributes(path.leaf).state_value
        while path_state is not path.root:
            path_state = path.get_predecessor(path_state)
            path_action = path.get_action(path_state)
            action_values = self.graph.get_attributes(path_state).action_values
            action_values[path_action] = subtree_value
            subtree_value = np.mean(action_values)
