from alpha_viergewinnt import data_logger


class DummyLogger():
    def __call__(self, value):
        self.value = value


def test_data_logger_set_logger_and_log():
    dummy_logger = DummyLogger()
    data_logger.set_logger(dummy_logger)
    data_logger.log(42)

    assert dummy_logger.value == 42
