import numpy as np


class GameNotFinishedException(Exception):
    pass


class Evaluator(object):
    # constants for state values and state array
    STATE_VALUE_WIN = 1
    STATE_VALUE_LOSS = -1
    STATE_VALUE_DRAW = 0

    STATE_ARRAY_PLAYER = 1
    STATE_ARRAY_OPPONENT = -1

    def __init__(self, estimator, player):
        self.estimator = estimator
        self.player = player

    def evaluate(self, state):
        try:
            state_value = self._get_final_state_value(state)
        except GameNotFinishedException:
            game_finished = False
            state_array = self._get_array_from_state(state)
            prior_distribution, state_value = self.estimator.infer(state_array)
            return prior_distribution, state_value, game_finished

        game_finished = True
        prior_distribution = np.zeros(len(self.estimator.actions))
        return prior_distribution, state_value, game_finished

    def _get_final_state_value(self, state):
        if state.is_winner(self.player):
            return self.STATE_VALUE_WIN
        elif state.is_winner(self.player.opponent()):
            return self.STATE_VALUE_LOSS
        elif state.is_draw():
            return self.STATE_VALUE_DRAW
        else:
            raise GameNotFinishedException()

    def _get_array_from_state(self, state):
        return state.get_array_view(
            player=self.player,
            player_value=self.STATE_ARRAY_PLAYER,
            opponent=self.player.opponent(),
            opponent_value=self.STATE_ARRAY_OPPONENT)

    def train(self, states_and_search_distributions, final_state):
        target_state_value = self._get_final_state_value(final_state)

        states, target_distributions = zip(*states_and_search_distributions)
        target_distribution_batch = np.stack(target_distributions)
        state_array_batch = np.stack([self._get_array_from_state(state) for state in states])
        target_state_value_batch = np.full(len(states_and_search_distributions), target_state_value)
        loss = self.estimator.train(state_array_batch, target_distribution_batch, target_state_value_batch)
        return loss
