import numpy as np


class Attributes(object):
    def __init__(self, state_value, prior_distribution):
        self.state_value = state_value
        self.prior_distribution = prior_distribution
        self.action_value = np.zeros(len(prior_distribution))
        self.visit_count = np.zeros(len(prior_distribution))

    def __str__(self):
        return 'state_value=' + _try_format_float(self.state_value) + '\nprior_distribution=' + _try_format_float(
            self.prior_distribution) + '\naction_value=' + _try_format_float(
                self.action_value) + '\nvisit_count=' + _try_format_int(self.visit_count)


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
