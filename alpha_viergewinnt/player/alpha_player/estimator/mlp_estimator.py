import numpy as np
from torch import tensor, sigmoid
from torch.nn import Linear, Module
from torch.nn.functional import softmax, cross_entropy, mse_loss
from torch.optim import SGD


class MlpEstimator(Module):
    def __init__(self, board_size, actions):
        super().__init__()

        self.actions = actions
        board_width, board_height = board_size
        self.state_size = board_width * board_height

        self.hidden_size = self.state_size * 2
        self.actions_size = len(actions)

        self.layer_input = Linear(self.state_size, self.hidden_size)
        self.layer_action_values = Linear(self.hidden_size, self.actions_size)
        self.layer_state_value = Linear(self.hidden_size, 1)

        self.optimizer = SGD(self.parameters(), lr=0.001, momentum=0.9)

    def infer(self, state_array):
        state_tensor = tensor(state_array).float().view(1, self.state_size)
        action_values_tensor, state_value_tensor = self.forward(state_tensor)
        action_values = action_values_tensor.view(-1).detach().numpy()
        state_value = state_value_tensor.view(-1).detach().item()
        return action_values, state_value

    def forward(self, input_):
        hidden = sigmoid(self.layer_input(input_))
        action_values = softmax(self.layer_action_values(hidden), dim=1)
        state_value = sigmoid(self.layer_state_value(hidden))
        return action_values, state_value

    def learn(self, state_array, selected_action, final_value):
        state_tensor = tensor(state_array).float().view(1, self.state_size)
        target_state_value = tensor(final_value).view(1, 1).float()
        action_value_index = tensor([index for index, action in enumerate(self.actions) if action == selected_action])

        self.optimizer.zero_grad()
        action_values, state_value = self.forward(state_tensor)
        state_value_loss = mse_loss(state_value, target_state_value)
        action_values_loss = cross_entropy(action_values, action_value_index)
        loss = state_value_loss + action_values_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()
