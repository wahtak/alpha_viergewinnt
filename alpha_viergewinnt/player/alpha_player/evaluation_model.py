import numpy as np


class ConditionEvaluationModel(object):
    def __init__(self, win_condition, loss_condition, draw_condition):
        self.win_condition = win_condition
        self.loss_condition = loss_condition
        self.draw_condition = draw_condition

    def get_action_priors_and_state_values(self, actions, state):
        action_priors = np.ones(len(actions)) / len(actions)

        state_value = 0
        if state.check(self.win_condition):
            state_value = 1
        if state.check(self.loss_condition):
            state_value = -1

        return action_priors, state_value


# def create_model(board, num_moves):
#     hidden_in = 42
#     output_of_input_layer = tf.contrib.layers.fully_connected(board, hidden_dim)
#     _output = tf.contrib.layers.fully_connected(board, hidden_dim)

# class AlphaStrategy(object):
#     def __init__(self, board_shape, input_enum, num_moves):
#         # self.input_offset = np.mean([element for element in input_enum])

#         self.board_placeholder = tf.placeholder(dtype=tf.int8, shape=board_shape, name='board')
#         self.move_probabilities, self.state_utility = create_model(self.board_placeholder, num_moves)
