import numpy as np

VALUE_TO_CHAR = {0: '.', 1: 'X', 2: 'O'}
OUTPUT_HEADER = [str(index) for index in range(1, 8)]
LINE_TERMINATE = '\n'
SEPERATOR = ' '


class Board(object):
    def __init__(self, size):
        self.field = np.zeros(size)

    def __str__(self):
        output_chars = []
        output_chars.extend([str(index + 1) for index in range(self.field.shape[1])])
        output_chars.append(LINE_TERMINATE)
        for row in self.field:
            output_chars.extend([VALUE_TO_CHAR[cell] for cell in row])
            output_chars.append(LINE_TERMINATE)
        return SEPERATOR + SEPERATOR.join(output_chars)


def test_board_output():
    board = Board(size=(6, 7))
    output_string_length = len(board.__str__())
    print(board)


def test_board_set_get():
    board = Board(size=(3, 3))
    board.field[1, 1] = 1
    assert board.field[1, 1] == 1
    assert board.field[1, 0] == 0
    print(board)


if __name__ == '__main__':
    test_set_get()