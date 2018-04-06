import re


class HumanPlayer(object):
    def __init__(self, **kwargs):
        pass

    def get_next_move(self, state):
        selected_move = None
        possible_moves = state.get_possible_moves()

        while selected_move not in possible_moves:
            human_input = input('select move %s: ' % possible_moves)
            try:
                selected_move = get_move_from_string(human_input)
            except ValueError:
                pass
        return selected_move


def get_move_from_string(string):
    move = tuple(int(match) for match in re.findall('[\d]+', string))
    if len(move) == 1:
        return move.pop()
    return move
