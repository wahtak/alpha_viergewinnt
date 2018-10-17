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

    def __init__(self, estimator, player, opponent, win_condition, loss_condition, draw_condition):
        self.estimator = estimator
        self.player = player
        self.opponent = opponent
        self.win_condition = win_condition
        self.loss_condition = loss_condition
        self.draw_condition = draw_condition

    def __call__(self, state):
        return self.get_prior_distribution_and_state_value(state)

    def get_prior_distribution_and_state_value(self, state):
        state_value = self._get_final_state_value(state)

        if state_value is not None:
            game_finished = True
            prior_distribution = np.zeros(len(self.estimator.actions))
            return prior_distribution, state_value, game_finished

        game_finished = False
        state_array = self._get_array_from_state(state)
        prior_distribution, state_value = self.estimator.infer(state_array)
        return prior_distribution, state_value, game_finished

    def _get_final_state_value(self, state):
        if state.check(self.win_condition):
            return self.STATE_VALUE_WIN
        elif state.check(self.loss_condition):
            return self.STATE_VALUE_LOSS
        elif state.check(self.draw_condition):
            return self.STATE_VALUE_DRAW
        else:
            # game not yet finished
            return None

    def _get_array_from_state(self, state):
        return state.get_array_view(
            player=self.player,
            player_value=self.STATE_ARRAY_PLAYER,
            opponent=self.opponent,
            opponent_value=self.STATE_ARRAY_OPPONENT)

    def train(self, states_and_search_distributions, final_state):
        target_state_value = self._get_final_state_value(final_state)

        if target_state_value is None:
            raise GameNotFinishedException()

        states, target_distributions = zip(*states_and_search_distributions)
        target_distribution_batch = np.stack(target_distributions)
        state_array_batch = np.stack([self._get_array_from_state(state) for state in states])
        target_state_value_batch = np.full(len(states_and_search_distributions), target_state_value)
        loss = self.estimator.train(state_array_batch, target_distribution_batch, target_state_value_batch)
        return loss
