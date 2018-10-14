import logging

import torch
from torch import tensor, sigmoid, clamp
from torch.nn import Linear, Module
from torch.nn.functional import mse_loss, cross_entropy, softmax
from torch.optim import SGD


class MlpEstimator(Module):
    # constants for state values and state array
    STATE_VALUE_WIN = 1
    STATE_VALUE_LOSS = -1
    STATE_VALUE_DRAW = 0

    STATE_ARRAY_PLAYER = 1
    STATE_ARRAY_OPPONENT = -1

    def __init__(self, board_size, actions, learning_rate=0.01, hidden_layer_scale=10, filename=None, **kwargs):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

        self.actions = actions
        board_width, board_height = board_size
        self.state_size = board_width * board_height
        self.filename = filename

        self.hidden_size = self.state_size * hidden_layer_scale
        self.actions_size = len(actions)

        self.layer_input = Linear(self.state_size, self.hidden_size)
        self.layer_action_value = Linear(self.hidden_size, self.actions_size)
        self.layer_state_value = Linear(self.hidden_size, 1)

        self.optimizer = SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def infer(self, state_array):
        state_tensor = tensor(state_array).float().view(1, self.state_size)
        action_value_tensor, state_value_tensor = self.forward(state_tensor)
        action_value = softmax(action_value_tensor, dim=1).view(-1).detach().numpy()
        state_value = clamp(state_value_tensor, min=-1, max=1).view(-1).detach().item()
        return action_value, state_value

    def forward(self, input_):
        hidden = sigmoid(self.layer_input(input_))
        action_value = self.layer_action_value(hidden)
        state_value = self.layer_state_value(hidden)
        return action_value, state_value

    def learn(self, state_array, selected_action, final_state_value):
        state_tensor = tensor(state_array).float().view(1, self.state_size)
        target_state_value = tensor(final_state_value).view(1, 1).float()
        action_value_index = tensor([index for index, action in enumerate(self.actions) if action == selected_action])

        self.optimizer.zero_grad()
        action_value, state_value = self.forward(state_tensor)
        state_value_loss = mse_loss(state_value, target_state_value)
        action_value_loss = cross_entropy(action_value, action_value_index)
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
