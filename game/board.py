import enum

import numpy as np


class Player(enum.Enum):
    X = 1
    O = 2


LINE_TERMINATE = '\n'
SEPERATOR = ''
CHARACTER_MAPPING = {0: '.', Player.X.value: Player.X.name, Player.O.value: Player.O.name}


class Board(object):
    '''
    Generic playing board with string representation and hash (of current state).

    Use numpy arrays for speed during state transitions and winning condition checking.
    '''

    def __init__(self, size):
        self.state = np.zeros(size, dtype=np.int16)

    def __str__(self):
        output_chars = []
        output_chars.extend([str(index) for index in range(self.state.shape[1])])
        output_chars.append(LINE_TERMINATE)
        for row in np.flip(self.state, axis=0):
            output_chars.extend([CHARACTER_MAPPING[cell] for cell in row])
            output_chars.append(LINE_TERMINATE)
        return SEPERATOR + SEPERATOR.join(output_chars)

    def __hash__(self):
        # only consider state for hash
        return hash(self.state.tobytes())

    def get_player_state(self, player):
        return (self.state == player.value).astype(np.int16)