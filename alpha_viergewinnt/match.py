from copy import deepcopy


class Match(object):
    def __init__(self, game, players, win_conditions, draw_condition, trainers=None):
        self.game = game
        self.players = players
        self.win_conditions = win_conditions
        self.draw_condition = draw_condition
        self.trainers = trainers

    def evaluate(self, iterations):
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
        self._learn(game)
        return self._get_result(game)

    def _is_game_finished(self, game):
        if game.check(self.draw_condition):
            return True
        for player in self.players:
            if game.check(self.win_conditions[player]):
                return True
        return False

    def _play_move(self, game, record_moves=False):
        print(game)
        current_player = game.active_player
        next_move = self.players[current_player].get_next_move(game)
        if record_moves:
            self.trainers[current_player].record(deepcopy(game), next_move)
        game.play_move(player=current_player, move=next_move)
        print('Player %s plays %s' % (current_player.name, next_move))
        return next_move

    def _learn(self, game):
        for trainer in self.trainers.values():
            trainer.learn(game)

    def _get_result(self, game):
        print(game)
        for player in self.players:
            if game.check(self.win_conditions[player]):
                print('Player %s wins!' % player.name)
                return player

        print('Draw!')
        return None
