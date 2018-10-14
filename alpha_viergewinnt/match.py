import logging
from copy import deepcopy

import numpy as np


class Match(object):
    def __init__(self, game, players, win_conditions, draw_condition):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.game = game
        self.players = players
        self.win_conditions = win_conditions
        self.draw_condition = draw_condition

    def compare(self, iterations):
        results = {player: 0 for player in self.players}
        results[None] = 0

        for iteration in range(iterations):
            winner = self.play()
            results[winner] += 1

        return results

    def play(self):
        game = deepcopy(self.game)
        while not self._is_game_finished(game):
            self._play_move(game)
        return self._get_result(game)

    def train(self):
        game = deepcopy(self.game)
        while not self._is_game_finished(game):
            self._play_move(game, record_moves=True)
        result = self._get_result(game)
        loss = self._learn(game)
        return result, loss

    def _is_game_finished(self, game):
        if game.check(self.draw_condition):
            return True
        for player in self.players:
            if game.check(self.win_conditions[player]):
                return True
        return False

    def _play_move(self, game, record_moves=False):
        self.logger.debug(game)
        current_player = game.active_player
        next_move = self.players[current_player].get_next_move(game)
        game.play_move(player=current_player, move=next_move)
        self.logger.debug('Player %s plays %s' % (current_player.name, next_move))
        return next_move

    def _learn(self, game):
        losses = []
        for player in self.players.values():
            loss = player.learn(game)
            losses.append(loss)
        mean_loss = np.mean(losses)
        self.logger.info('Training loss %.4f' % mean_loss)
        return mean_loss

    def _get_result(self, game):
        self.logger.debug(game)
        for player in self.players:
            if game.check(self.win_conditions[player]):
                self.logger.debug('Player %s wins!' % player.name)
                return player

        self.logger.debug('Draw!')
        return None
