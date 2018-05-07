from copy import deepcopy


class AlreadyExpandedException(Exception):
    pass


class TreeSearch(object):
    def __init__(self, tree, selection_strategy, evaluation_model):
        self.tree = tree
        self.selection_strategy = selection_strategy
        self.evaluation_model = evaluation_model

    def select_leaf(self, source):
        state = source
        while self.tree.has_successors(state):
            actions, attributes = self._get_actions_and_attributes(state)
            selected_action = self.selection_strategy(actions, attributes)
            self.tree.get_transition_attributes(state, selected_action).visit_count += 1
            state = self.tree.get_successor(state, selected_action)
        return state

    def _get_actions_and_attributes(self, state):
        actions = self.tree.get_actions(state)
        attributes = [self.tree.get_transition_attributes(state, action) for action in actions]
        return actions, attributes

    def expand(self, leaf):
        if self.tree.has_successors(leaf):
            raise AlreadyExpandedException()

        possible_actions = leaf.get_possible_moves()
        for action in possible_actions:
            successor = deepcopy(leaf)
            successor.play_move(player=leaf.current_player, move=action)
            self.tree.add_successor(successor, source=leaf, action=action)

    def evaluate(self, state):
        actions, attributes = self._get_actions_and_attributes(state)
        prior_probabilities, state_value = self.evaluation_model(actions, state)
        for action, prior_probability in zip(actions, prior_probabilities):
            self.tree.get_transition_attributes(state, action).prior_probability = prior_probability
        self.tree.get_state_attributes(state).state_value = state_value

    def backup(self, leaf):
        pass
