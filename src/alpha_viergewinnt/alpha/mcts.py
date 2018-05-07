from copy import deepcopy


class AlreadyExpandedException(Exception):
    pass


class Mcts(object):
    def __init__(self, graph, selection_strategy, evaluation_model):
        self.graph = graph
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model

    def select_leaf(self, source):
        state = source
        while self.graph.has_successors(state):
            actions, attributes = self._get_actions_and_attributes(state)
            selected_action = self.selection_strategy(actions, attributes)
            self.graph.get_transition_attributes(state, selected_action).visit_count += 1
            state = self.graph.get_successor(state, selected_action)
        return state

    def _get_actions_and_attributes(self, state):
        actions = self.graph.get_actions(state)
        attributes = [self.graph.get_transition_attributes(state, action) for action in actions]
        return actions, attributes

    def expand(self, leaf):
        if self.graph.has_successors(leaf):
            raise AlreadyExpandedException()

        possible_actions = leaf.get_possible_moves()
        for action in possible_actions:
            successor = deepcopy(leaf)
            successor.play_move(player=leaf.current_player, move=action)
            self.graph.add_successor(successor, source=leaf, action=action)

    def evaluate(self, state):
        actions, attributes = self._get_actions_and_attributes(state)
        prior_probabilities, state_value = self.evaluation_model(actions, state)
        for action, prior_probability in zip(actions, prior_probabilities):
            self.graph.get_transition_attributes(state, action).prior_probability = prior_probability
        self.graph.get_state_attributes(state).state_value = state_value

    def backup(self, leaf):
        pass
