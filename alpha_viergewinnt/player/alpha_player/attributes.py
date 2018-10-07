import numpy as np


class Attributes(object):
    def __init__(self, state_value, prior_probabilities):
        self.state_value = state_value
        self.prior_probabilities = prior_probabilities
        self.action_values = np.zeros(len(prior_probabilities))
        self.visit_counts = np.zeros(len(prior_probabilities))

    def __str__(self):
        return 'state_value=' + _try_format_float(self.state_value) + '\nprior_probabilities=' + _try_format_float(
            self.prior_probabilities) + '\naction_values=' + _try_format_float(
                self.action_values) + '\nvisit_counts=' + _try_format_int(self.visit_counts)


def _try_format_float(value):
    try:
        return '%.02f' % value
    except TypeError:
        return '%s' % value


def _try_format_int(value):
    try:
        return '%d' % value
    except TypeError:
        return '%s' % value
