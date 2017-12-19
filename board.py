import enum

import numpy as np


class Stone(enum.Enum):
    X = 1
    O = 2


VALUE_TO_CHAR = {0: '.', Stone.X.value: Stone.X.name, Stone.O.value: Stone.O.name}
OUTPUT_HEADER = [str(index) for index in range(7)]
LINE_TERMINATE = '\n'
SEPERATOR = ' '


class Board(object):
    '''Generic playing board with string representation.'''

    def __init__(self, size):
        self.field = np.zeros(size, dtype=np.int16)

    def __str__(self):
        output_chars = []
        output_chars.extend([str(index + 1) for index in range(self.field.shape[1])])
        output_chars.append(LINE_TERMINATE)
        for row in np.flip(self.field, axis=0):
            output_chars.extend([VALUE_TO_CHAR[cell] for cell in row])
            output_chars.append(LINE_TERMINATE)
        return SEPERATOR + SEPERATOR.join(output_chars)


class IllegalMoveException(Exception):
    pass


class ViergewinntBoard(Board):
    '''Playing board for vier gewinnt.'''

    def __init__(self, *args, **kwargs):
        super(ViergewinntBoard, self).__init__(*args, **kwargs)

    def insert(self, stone, column):
        try:
            column_vector = self.field.T[column]
        except IndexError:
            raise IllegalMoveException('column %d does not exist.' % column)

        for index, element in enumerate(column_vector):
            if element == 0:
                column_vector[index] = stone.value
                break
        else:
            raise IllegalMoveException('column %d is full' % column)


def test_board_output():
    board = Board(size=(6, 7))
    output_string_length = len(board.__str__())
    print(board)


def test_board_set_get():
    board = Board(size=(3, 3))
    board.field[0, 1] = 1
    assert board.field[0, 1] == 1
    print(board)


def test_viergewinnt_board_insert():
    import pytest
    board = ViergewinntBoard(size=(6, 7))

    # valid insert
    board.insert(stone=Stone.X, column=1)
    assert board.field[0, 1] == Stone.X.value

    # column does not exist
    with pytest.raises(IllegalMoveException):
        board.insert(stone=Stone.X, column=7)

    # column full
    for _ in range(6):
        board.insert(stone=Stone.X, column=2)
    with pytest.raises(IllegalMoveException):
        board.insert(stone=Stone.X, column=2)


if __name__ == '__main__':
    test_viergewinnt_board_insert()