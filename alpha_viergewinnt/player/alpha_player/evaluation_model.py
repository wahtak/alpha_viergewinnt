import numpy as np

VALUE_WIN = 1
VALUE_LOSS = -1
VALUE_DRAW = 0


class GameNotFinishedException(Exception):
    pass


class EvaluationModel(object):
    def __init__(self, estimator, player, opponent, win_condition, loss_condition, draw_condition):
        self.estimator = estimator
        self.player = player
        self.opponent = opponent
        self.win_condition = win_condition
        self.loss_condition = loss_condition
        self.draw_condition = draw_condition

    def __call__(self, actions, state):
        return self.get_prior_probabilities_and_state_value(actions, state)

    def get_prior_probabilities_and_state_value(self, actions, state):
        state_value = self._get_final_state_value(state)

        if state_value is not None:
            game_finished = True
            prior_probabilities = np.zeros(len(actions))
            return prior_probabilities, state_value, game_finished

        game_finished = False
        state_array = self._get_array_from_state(state)
        all_likelihoods, state_value = self.estimator.infer(state_array)
        prior_probabilities = self._get_probabilities_for_possible_actions(
            all_likelihoods=all_likelihoods, all_actions=self.estimator.actions, possbile_actions=actions)
        return prior_probabilities, state_value, game_finished

    def _get_final_state_value(self, state):
        if state.check(self.win_condition):
            return VALUE_WIN
        elif state.check(self.loss_condition):
            return VALUE_LOSS
        elif state.check(self.draw_condition):
            return VALUE_DRAW
        else:
            # game not yet finished
            return None

    def _get_array_from_state(self, state):
        return state.get_array_view(player=self.player, player_value=1, opponent=self.opponent, opponent_value=-1)

    def _get_probabilities_for_possible_actions(self, all_likelihoods, all_actions, possbile_actions):
        likelihoods = [value for action, value in zip(all_actions, all_likelihoods) if action in possbile_actions]
        probabilities = likelihoods / np.sum(likelihoods)
        return probabilities

    def learn(self, states_and_selected_actions, final_state):
        final_state_value = self._get_final_state_value(final_state)

        if final_state_value is None:
            raise GameNotFinishedException()

        losses = []
        for state, selected_action in states_and_selected_actions:
            state_array = self._get_array_from_state(state)
            loss = self.estimator.learn(state_array, selected_action, final_state_value)
            losses.append(loss)

        return np.mean(losses)
