import enum

import numpy as np


class Player(enum.Enum):
    X = 1
    O = 2  # noqa E741

    def opponent(self):
        return Player.X if self == Player.O else Player.O


class RowOrder(enum.Enum):
    NORMAL = 1
    REVERSED = 2


CORNER = " "
LINE_TERMINATE = "\n"
SEPERATOR = ""
CHARACTER_MAPPING = {0: ".", Player.X.value: Player.X.name, Player.O.value: Player.O.name}


class Board(object):
    """
    Generic playing board with string representation and hash (of current state).

    Use numpy arrays for speed during state transitions and condition checking.
    """

    def __init__(self, size, output_row_order):
        self.state = np.zeros(size, dtype=np.int16)
        self.output_row_order = output_row_order

    def _row_iter(self):
        row_indices = [str(row_index) for row_index in range(self.state.shape[0])]

        if self.output_row_order == RowOrder.REVERSED:
            row_iter = zip(reversed(row_indices), np.flip(self.state, axis=0))
        else:
            row_iter = zip(row_indices, self.state)

        return row_iter

    def __str__(self):
        output_chars = []
        output_chars.append(CORNER)
        output_chars.extend([str(column_index) for column_index in range(self.state.shape[1])])
        output_chars.append(LINE_TERMINATE)
        for row_index, row in self._row_iter():
            output_chars.append(row_index)
            output_chars.extend([CHARACTER_MAPPING[cell] for cell in row])
            output_chars.append(LINE_TERMINATE)
        return SEPERATOR + SEPERATOR.join(output_chars)

    def __hash__(self):
        # only consider state for hash
        return hash(self.state.tobytes())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_array_view(self, player, player_value, opponent=None, opponent_value=0):
        """
        Get board state as an array, where stones of player have the value player_value and stones
        of the opponent (optional) have the value opponent_value.
        """
        player_array = (self.state == player.value).astype(np.int16) * player_value

        if opponent is not None:
            opponent_array = (self.state == opponent.value).astype(np.int16) * opponent_value
            return player_array + opponent_array
        else:
            return player_array
