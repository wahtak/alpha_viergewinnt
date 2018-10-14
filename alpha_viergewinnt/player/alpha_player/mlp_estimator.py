import logging

import torch
from torch import tensor, tanh, sum, mean, log
from torch.nn import Linear, Module
from torch.nn.functional import mse_loss, softmax, relu
from torch.optim import Adam


class MlpEstimator(Module):
    # constants for state values and state array
    STATE_VALUE_WIN = 1
    STATE_VALUE_LOSS = -1
    STATE_VALUE_DRAW = 0

    STATE_ARRAY_PLAYER = 1
    STATE_ARRAY_OPPONENT = -1

    def __init__(self, board_size, actions, hidden_layer_scale=10, filename=None, **kwargs):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        self.actions = actions
        board_width, board_height = board_size
        self.state_size = board_width * board_height
        self.filename = filename

        self.hidden_size = self.state_size * hidden_layer_scale
        self.action_size = len(actions)

        self.layer_input = Linear(self.state_size, self.hidden_size)
        self.layer_hidden1 = Linear(self.hidden_size, self.hidden_size)
        self.layer_hidden2 = Linear(self.hidden_size, self.hidden_size)
        self.layer_hidden3 = Linear(self.hidden_size, self.hidden_size)
        self.layer_action_distribution = Linear(self.hidden_size, self.action_size)
        self.layer_state_value = Linear(self.hidden_size, 1)

        self.optimizer = Adam(self.parameters())

    def infer(self, state_array):
        state = tensor(state_array).float().view(1, self.state_size)
        action_distribution, state_value = self.forward(state)
        action_distribution_array = action_distribution.view(-1).detach().numpy()
        state_value_array = state_value.view(-1).detach().item()
        return action_distribution_array, state_value_array

    def forward(self, state):
        input_ = relu(self.layer_input(state))
        hidden1 = relu(self.layer_hidden1(input_))
        hidden2 = relu(self.layer_hidden2(hidden1))
        hidden3 = relu(self.layer_hidden3(hidden2))
        # action_distribution = normalize(sigmoid(self.layer_action(hidden)), p=1, dim=1)
        action_distribution = softmax(self.layer_action_distribution(hidden3), dim=1)
        state_value = tanh(self.layer_state_value(hidden3))
        return action_distribution, state_value

    def learn(self, state_array, target_distribution_array, target_state_value_array):
        state = tensor(state_array).float().view(1, self.state_size)
        target_state_value = tensor(target_state_value_array).view(1, 1).float()
        target_distribution = tensor(target_distribution_array).float().view(1, self.action_size)

        self.optimizer.zero_grad()
        action_distribution, state_value = self.forward(state)
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
