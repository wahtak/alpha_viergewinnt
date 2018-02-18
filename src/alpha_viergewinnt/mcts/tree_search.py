'''Functions for running MCTS on a GameTree.

Chaslot, Guillaume & Winands, Mark & Herik, H & Uiterwijk, Jos & Bouzy, Bruno. (2008).
Progressive Strategies for Monte-Carlo Tree Search. New Mathematics and Natural Computation.
04. 343-357. 10.1142/S1793005708001094.'''

from copy import deepcopy


class NoUnexploredMovesException(Exception):
    pass


class TreeSearch(object):
    def __init__(self, tree, selection_strategy, expansion_strategy):
        self.tree = tree
        self.selection_strategy = selection_strategy
        self.expansion_strategy = expansion_strategy

    def select_leaf(self, source):
        state = source
        while not self._is_leaf(state):
            explored_moves = self.tree.get_transitions(state)
            selected_move = self.selection_strategy(explored_moves)
            state = self.tree.get_successor(state, selected_move)
        return state

    def _is_leaf(self, state):
        has_unexplored_moves = len(self._get_unexplored_moves(state)) > 0
        no_moves_possible = len(state.get_possible_moves()) == 0
        return has_unexplored_moves or no_moves_possible

    def _get_unexplored_moves(self, state):
        explored_moves = self.tree.get_transitions(state)
        possible_moves = state.get_possible_moves()
        return set(possible_moves) - set(explored_moves)

    def expand(self, source):
        unexplored_moves = self._get_unexplored_moves(source)
        if len(unexplored_moves) == 0:
            raise NoUnexploredMovesException()
        selected_move = self.expansion_strategy(unexplored_moves)
        successor = deepcopy(source)
        successor.play_move(player=source.current_player, move=selected_move)
        self.tree.add_successor(source=source, transition=selected_move, successor=successor)
        return successor

    def backpropagate(self, source, delta_weight):
        for state in self.tree.get_path_to_root(source):
            self.tree.attributes[state].visit_count += 1
            self.tree.attributes[state].weight += delta_weight


class Simulator(object):
    def __init__(self, strategy, win_condition, loss_condition, draw_condition):
        self.strategy = strategy
        self.win_condition = win_condition
        self.loss_condition = loss_condition
        self.draw_condition = draw_condition

    def _is_final_state(self, state):
        return state.check(self.win_condition) or state.check(self.loss_condition) or state.check(self.draw_condition)

    def get_rollout_value(self, state):
        assert self._is_final_state(state) is True

        rollout_value = 0
        if state.check(self.win_condition):
            rollout_value += 1
        if state.check(self.loss_condition):
            rollout_value -= 1
        return rollout_value

    def rollout(self, initial_state):
        state = initial_state
        while not self._is_final_state(state):
            possible_moves = state.get_possible_moves()
            selected_move = self.strategy(possible_moves)
            state.play_move(state.current_player, selected_move)
        return state
