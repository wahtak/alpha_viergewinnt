import logging
from copy import deepcopy

import numpy as np


class Match(object):
    def __init__(self, game, players):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.game = game
        self.players = players

    def _is_game_finished(self, game):
        if game.is_draw():
            return True
        for player in self.players:
            if game.is_winner(player):
                return True
        return False

    def _play_move(self, game, record_moves=False):
        self.logger.debug(game)
        current_player = game.active_player
        next_move = self.players[current_player].get_next_move(game)
        game.play_move(player=current_player, move=next_move)
        self.logger.debug('Player %s plays %s' % (current_player.name, next_move))
        return next_move


class ComparisonMatch(Match):
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

    def _get_result(self, game):
        self.logger.debug(game)
        for player in self.players:
            if game.is_winner(player):
                self.logger.debug('Player %s wins!' % player.name)
                return player

        self.logger.debug('Draw!')
        return None


class TrainingMatch(Match):
    def train(self):
        game = deepcopy(self.game)
        while not self._is_game_finished(game):
            self._play_move(game, record_moves=True)
        loss = self._train(game)
        return loss

    def _train(self, game):
        losses = []
        for player in self.players.values():
            loss = player.train(game)
            losses.append(loss)
        mean_loss = np.mean(losses)
        self.logger.info('Training loss %.4f' % mean_loss)
        return mean_loss
