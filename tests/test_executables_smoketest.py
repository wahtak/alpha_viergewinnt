from subprocess import run

import pytest


@pytest.mark.skip()
def test_train_smoketest():
    command = [
        'bin/train', '--game=viergewinnt', '--estimator=mlp', '--mcts-steps=3', '--training-iterations=3',
        '--comparison-iterations=3'
    ]
    assert run(command).returncode == 0


@pytest.mark.skip()
def test_play_smoketest():
    command = ['bin/play', '--game=viergewinnt', '-x', 'random', '-o', 'random']
    assert run(command).returncode == 0
