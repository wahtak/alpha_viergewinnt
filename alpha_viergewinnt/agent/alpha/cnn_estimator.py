import logging

import torch
from torch import tensor, tanh, sum, mean, log
from torch.nn import Linear, Module, Conv2d
from torch.nn.functional import mse_loss, softmax, relu
from torch.optim import Adam


class CnnEstimator(Module):
    def __init__(
        self,
        board_size,
        actions,
        hidden_layer_scale=10,
        num_common_hidden_layers=3,
        num_action_hidden_layers=2,
        num_value_hidden_layers=2,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)

        self.actions = actions
        self.board_size = board_size
        board_width, board_height = board_size
        self.input_size = board_width * board_height
        # self.hidden_size = board_size * 10

        conv_kernel_size = 4
        conv_num_channels = 64
        self.conv_output_size = (
            conv_num_channels * (board_width - conv_kernel_size + 1) * (board_height - conv_kernel_size + 1)
        )
        self.hidden_size = self.input_size * 10

        self.action_size = len(actions)
        self.value_size = 1

        # input linear layers
        # self.fc_input = Linear(self.input_size, self.hidden_size)

        # common conv layers
        self.cv_common1 = Conv2d(in_channels=1, out_channels=conv_num_channels, kernel_size=conv_kernel_size)

        # common linear layers
        self.fc_common2 = Linear(self.conv_output_size, self.conv_output_size)
        self.fc_common3 = Linear(self.conv_output_size, self.hidden_size)

        # action distribution layers
        self.fc_action1 = Linear(self.hidden_size, self.hidden_size)
        self.fc_action2 = Linear(self.hidden_size, self.action_size)

        # state value layers
        self.fc_value1 = Linear(self.hidden_size, self.hidden_size)
        self.fc_value2 = Linear(self.hidden_size, self.value_size)
        self.optimizer = Adam(self.parameters(), weight_decay=5e-4)

    def infer(self, state_array):
        state = tensor(state_array).float().view(1, *self.board_size)
        action_distribution, state_value = self._forward(state)
        action_distribution_array = action_distribution.view(-1).detach().numpy()
        state_value_array = state_value.view(-1).detach().item()
        return action_distribution_array, state_value_array

    def _forward(self, state):
        # common
        common_hidden = state.view(-1, 1, *self.board_size)
        common_hidden = relu(self.cv_common1(common_hidden))
        common_hidden = common_hidden.view(-1, self.conv_output_size)
        common_hidden = relu(self.fc_common2(common_hidden))
        common_hidden = relu(self.fc_common3(common_hidden))

        # action_distribution
        action_hidden = relu(self.fc_action1(common_hidden))
        action_distribution = softmax(self.fc_action2(action_hidden), dim=1)
        # action_distribution = normalize(sigmoid(self.fc_distr2(distr)), p=1, dim=1)

        # state_value
        value_hidden = relu(self.fc_value1(common_hidden))
        state_value = tanh(self.fc_value2(value_hidden))

        return action_distribution, state_value

    def train(self, state_array, target_distribution_array, target_state_value_array):
        state = tensor(state_array).float().view(-1, *self.board_size)
        target_state_value = tensor(target_state_value_array).float().view(-1, self.value_size)
        target_distribution = tensor(target_distribution_array).float().view(-1, self.action_size)

        self.optimizer.zero_grad()
        action_distribution, state_value = self._forward(state)
        state_value_loss = mse_loss(state_value, target_state_value)
        action_value_loss = mean(sum(target_distribution * -log(action_distribution), dim=1))
        loss = state_value_loss + action_value_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, filename)
        self.logger.info("Saved parameters to %s" % filename)

    def load(self, filename):
        try:
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)
            self.logger.info("Loaded parameters from %s" % filename)
        except FileNotFoundError:
            self.logger.warning("Could not load parameters from %s" % filename)
