import logging

import torch
from torch import tensor, tanh, sum, mean, log
from torch.nn import Linear, Module, BatchNorm1d
from torch.nn.functional import mse_loss, softmax, relu
from torch.optim import Adam


class MlpEstimator(Module):
    def __init__(self, board_size, actions, hidden_layer_scale=10, filename=None):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        self.actions = actions
        board_width, board_height = board_size
        self.input_size = board_width * board_height
        self.filename = filename

        self.hidden_size = self.input_size * hidden_layer_scale
        self.distr_size = len(actions)
        self.value_size = 1

        # common hidden layers
        self.fc_common1 = Linear(self.input_size, self.hidden_size)
        self.bn_common1 = BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.fc_common2 = Linear(self.hidden_size, self.hidden_size)
        self.bn_common2 = BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.fc_common3 = Linear(self.hidden_size, self.hidden_size)
        self.bn_common3 = BatchNorm1d(self.hidden_size, track_running_stats=False)

        # action distribution layers
        self.fc_distr1 = Linear(self.hidden_size, self.hidden_size)
        self.bn_distr1 = BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.fc_distr2 = Linear(self.hidden_size, self.distr_size)

        # state value layers
        self.fc_value1 = Linear(self.hidden_size, self.hidden_size)
        self.bn_value1 = BatchNorm1d(self.hidden_size, track_running_stats=False)
        self.fc_value2 = Linear(self.hidden_size, self.value_size)

        self.optimizer = Adam(self.parameters(), weight_decay=5e-4)

    def infer(self, state_array):
        state = tensor(state_array).float().view(1, self.input_size)
        action_distribution, state_value = self._forward(state)
        action_distribution_array = action_distribution.view(-1).detach().numpy()
        state_value_array = state_value.view(-1).detach().item()
        return action_distribution_array, state_value_array

    def _forward(self, state):
        # common
        common = relu(self.fc_common1(state))
        common = relu(self.fc_common2(common))
        common = relu(self.fc_common3(common))

        # action_distribution
        distr = relu(self.fc_distr1(common))
        action_distribution = softmax(self.fc_distr2(distr), dim=1)
        # action_distribution = normalize(sigmoid(self.fc_distr2(distr)), p=1, dim=1)

        # state_value
        value = relu(self.fc_value1(common))
        state_value = tanh(self.fc_value2(value))

        return action_distribution, state_value

    def train(self, state_array, target_distribution_array, target_state_value_array):
        state = tensor(state_array).float().view(-1, self.input_size)
        target_state_value = tensor(target_state_value_array).float().view(-1, self.value_size)
        target_distribution = tensor(target_distribution_array).float().view(-1, self.distr_size)

        self.optimizer.zero_grad()
        action_distribution, state_value = self._forward(state)
        state_value_loss = mse_loss(state_value, target_state_value)
        action_value_loss = mean(sum(target_distribution * -log(action_distribution), dim=1))
        loss = state_value_loss + action_value_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.filename)
        self.logger.info('Saved parameters to %s' % self.filename)

    def load(self):
        try:
            state_dict = torch.load(self.filename)
            self.load_state_dict(state_dict)
            self.logger.info('Loaded parameters from %s' % self.filename)
        except FileNotFoundError:
            self.logger.warning('Could not load parameters from %s' % self.filename)
