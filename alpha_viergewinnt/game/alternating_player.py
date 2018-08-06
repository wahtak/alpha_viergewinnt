from .board import Player


class NotPlayersTurnException(Exception):
    pass


class AlternatingPlayer(object):
    '''Functionality for checking alternating player turns'''

    def __init__(self, starting_player):
        self.current_player = starting_player

    def register_player_turn(self, player):
        if player != self.current_player:
            raise NotPlayersTurnException('not player %s\'s turn' % player)
        self.current_player = Player.O if self.current_player == Player.X else Player.X

    def get_state_from_current_player_perspective(self):
        return self.get_state_from_player_perspective(self.current_player)
