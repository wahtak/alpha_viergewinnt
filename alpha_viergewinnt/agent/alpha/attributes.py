import numpy as np


class Attributes(object):
    def __init__(self, state_value, prior_distribution):
        self.state_value = state_value
        self.prior_distribution = prior_distribution
        self.action_value = np.zeros(len(prior_distribution))
        self.visit_count = np.zeros(len(prior_distribution))

    def __str__(self):
        return "state_value=%.2f\nprior_distribution=%s\naction_value=%s\nvisit_count=%s" % (
            self.state_value,
            _try_format_float_array(self.prior_distribution),
            _try_format_float_array(self.action_value),
            _try_format_float_array(self.visit_count),
        )


def _try_format_float_array(array):
    try:
        return np.array_str(array, precision=2, suppress_small=True)
    except TypeError:
        return str(array)


def _try_format_int_array(array):
    try:
        return np.array_str(array)
    except TypeError:
        return str(array)
