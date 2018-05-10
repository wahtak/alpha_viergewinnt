class StateAttributes(object):
    def __init__(self, state_value=None):
        self.state_value = state_value

    def __str__(self):
        return 'state_value=' + _try_format_float(self.state_value)


class ActionAttributes(object):
    def __init__(self, action_value=None, prior_probability=None, visit_count=0):
        self.action_value = action_value
        self.prior_probability = prior_probability
        self.visit_count = visit_count

    def __str__(self):
        return 'action_value=' + _try_format_float(self.action_value) + '\nprior_probability=' + _try_format_float(
            self.prior_probability) + '\nvisit_count=' + _try_format_int(self.visit_count)


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
