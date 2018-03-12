class HumanPlayer(object):
    def __init__(self, **kwargs):
        pass

    def get_next_move(self, state):
        selected_move = None
        while selected_move not in state.get_possible_moves():
            human_input = input('select move %s: ' % state.get_possible_moves())
            try:
                selected_move = int(human_input)
            except ValueError:
                pass
        return selected_move
