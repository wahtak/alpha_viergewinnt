class AlphaTrainer(object):
    def __init__(self, evaluation_model):
        self.evaluation_model = evaluation_model
        self.states_and_selected_actions = []

    def record(self, state, selected_action):
        self.states_and_selected_actions.append((state, selected_action))

    def learn(self, final_state):
        self.evaluation_model.learn(self.states_and_selected_actions, final_state)
