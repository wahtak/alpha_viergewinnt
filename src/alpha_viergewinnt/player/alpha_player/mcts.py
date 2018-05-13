from copy import deepcopy

import numpy as np


class AlreadyExpandedException(Exception):
    pass


class Mcts(object):
    def __init__(self, graph, path_factory, selection_strategy, evaluation_model):
        self.graph = graph
        self.path_factory = path_factory
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model

    def select(self, source):
        path = self.path_factory(source)
        state = source
        while self.graph.has_successors(state):
            actions = self.graph.get_actions(state)
            attributes = [self.graph.get_action_attributes(state, action) for action in actions]
            selected_action = self.selection_strategy(actions, attributes)
            self.graph.get_action_attributes(state, selected_action).visit_count += 1
            successor = self.graph.get_successor(state, selected_action)
            path.add_successor(successor, selected_action)
            state = successor
        return path

    def expand(self, leaf):
        if self.graph.has_successors(leaf):
            raise AlreadyExpandedException()

        possible_actions = leaf.get_possible_moves()
        for action in possible_actions:
            successor = deepcopy(leaf)
            successor.play_move(player=leaf.current_player, move=action)
            self.graph.add_successor(successor, source=leaf, action=action)

    def evaluate(self, state):
        actions = self.graph.get_actions(state)
        prior_probabilities, state_value = self.evaluation_model(actions, state)
        for action, prior_probability in zip(actions, prior_probabilities):
            self.graph.get_action_attributes(state, action).prior_probability = prior_probability
        self.graph.get_state_attributes(state).state_value = state_value

    def backup(self, path):
        path_state = path.leaf
        subtree_value = self.graph.get_state_attributes(path.leaf).state_value
        while path_state is not path.root:
            path_state = path.get_predecessor(path_state)
            path_action = path.get_action(path_state)
            self.graph.get_action_attributes(path_state, path_action).action_value = subtree_value
            actions = self.graph.get_actions(path_state)
            action_values = [self.graph.get_action_attributes(path_state, action).action_value for action in actions]
            subtree_value = np.mean(action_values)
