from .board import Player


class NotPlayersTurnException(Exception):
    pass


class AlternatingPlayer(object):
    """Functionality for checking alternating player turns"""

    def __init__(self, starting_player):
        self.active_player = starting_player
        self.idle_player = Player.opponent(starting_player)

    def register_player_turn(self, player):
        if player != self.active_player:
            raise NotPlayersTurnException("not player %s's turn" % player)
        self.idle_player = self.active_player
        self.active_player = Player.opponent(self.active_player)

    def get_state_from_active_player_perspective(self):
        return self.get_state_from_player_perspective(self.active_player)


class MoveRecorder(object):
    def __init__(self):
        self.recorded_moves = tuple()

    def record_move(self, move):
        self.recorded_moves += (move,)

    def __hash__(self):
        return hash(self.recorded_moves)
