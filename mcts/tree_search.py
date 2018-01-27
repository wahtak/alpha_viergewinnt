'''Functions for running MCTS on a GameTree.

Chaslot, Guillaume & Winands, Mark & Herik, H & Uiterwijk, Jos & Bouzy, Bruno. (2008).
Progressive Strategies for Monte-Carlo Tree Search. New Mathematics and Natural Computation.
04. 343-357. 10.1142/S1793005708001094.'''

import random
from copy import deepcopy


class NoUnexploredMovesException(Exception):
    pass


def select_leaf(game_tree, state, selection_strategy):
    def leaf_condition(game_tree, state):
        explored_moves = game_tree.get_successors(state).keys()
        possible_moves = state.get_possible_moves()
        return len(possible_moves) == 0 or set(explored_moves) != set(possible_moves)

    while not leaf_condition(game_tree, state):
        successors = game_tree.get_successors(state)
        selected_move = selection_strategy(successors.keys())
        state = successors[selected_move]
    return state


def expand_tree(game_tree, state, expansion_strategy):
    current_player = state.current_player
    explored_moves = game_tree.get_successors(state).keys()
    unexplored_moves = set(state.get_possible_moves()) - set(explored_moves)
    if len(unexplored_moves) == 0:
        raise NoUnexploredMovesException()

    selected_move = expansion_strategy(unexplored_moves)
    new_state = deepcopy(state)
    new_state.play_move(player=current_player, move=selected_move)
    game_tree.add_successor(state=state, move=selected_move, new_state=new_state)
    return new_state


def simulate_until_end(state, end_condition, simulation_strategy):
    state = deepcopy(state)

    while not end_condition(state):
        end_condition_eval = end_condition(state)
        possible_moves = state.get_possible_moves()
        selected_move = simulation_strategy(possible_moves)
        state.play_move(state.current_player, selected_move)
    return state


def backpropagate(game_tree, state, delta_weight):
    for state in game_tree.get_ancestors(state) | {state}:
        game_tree.attributes[state].visit_count += 1
        game_tree.attributes[state].weight += delta_weight