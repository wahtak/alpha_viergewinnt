import logging

import torch
from torch import tensor, tanh, sum, mean, log
from torch.nn import Linear, Module
from torch.nn.functional import mse_loss, softmax, relu
from torch.optim import Adam


class MlpEstimator(Module):
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
        board_width, board_height = board_size
        self.input_size = board_width * board_height
        self.num_common_hidden_layers = num_common_hidden_layers
        self.num_action_hidden_layers = num_action_hidden_layers
        self.num_value_hidden_layers = num_value_hidden_layers

        hidden_size = self.input_size * hidden_layer_scale
        self.action_size = len(actions)
        self.value_size = 1

        # common hidden layers
        self.fc_common_input = Linear(self.input_size, hidden_size)
        self.fc_common_hidden = [Linear(hidden_size, hidden_size) for _ in range(self.num_common_hidden_layers)]

        # action distribution layers
        self.fc_action_hidden = [Linear(hidden_size, hidden_size) for _ in range(self.num_action_hidden_layers)]
        self.fc_action_output = Linear(hidden_size, self.action_size)

        # state value layers
        self.fc_value_hidden = [Linear(hidden_size, hidden_size) for _ in range(self.num_value_hidden_layers)]
        self.fc_value_output = Linear(hidden_size, self.value_size)

        self.optimizer = Adam(self.parameters(), weight_decay=5e-4)

    def infer(self, state_array):
        state = tensor(state_array).float().view(1, self.input_size)
        action_distribution, state_value = self._forward(state)
        action_distribution_array = action_distribution.view(-1).detach().numpy()
        state_value_array = state_value.view(-1).detach().item()
        return action_distribution_array, state_value_array

    def _forward(self, state):
        # common
        common_hidden = relu(self.fc_common_input(state))
        for layer in self.fc_common_hidden:
            common_hidden = relu(layer(common_hidden))

        # action_distribution
        action_hidden = common_hidden
        for layer in self.fc_action_hidden:
            action_hidden = relu(layer(action_hidden))
        action_distribution = softmax(self.fc_action_output(action_hidden), dim=1)
        # action_distribution = normalize(sigmoid(self.fc_distr2(distr)), p=1, dim=1)

        # state_value
        value_hidden = common_hidden
        for layer in self.fc_value_hidden:
            value_hidden = relu(layer(value_hidden))
        state_value = tanh(self.fc_value_output(value_hidden))

        return action_distribution, state_value

    def train(self, state_array, target_distribution_array, target_state_value_array):
        state = tensor(state_array).float().view(-1, self.input_size)
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
