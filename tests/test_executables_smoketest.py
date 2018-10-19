from subprocess import run

import pytest


# @pytest.mark.skip()
@pytest.mark.parametrize('game,estimator', [('viergewinnt', 'generic'), ('tictactoe', 'mlp'), ('viergewinnt', 'mlp')])
def test_train_smoketest(game, estimator):
    command = [
        'bin/train', '--game', game, '--estimator', estimator, '--mcts-steps=3', '--num-training-games=3',
        '--num-comparison-games=3', '--disable-plotting=True', '--num-epochs=3'
    ]
    assert run(command).returncode == 0


# @pytest.mark.skip()
@pytest.mark.parametrize('game,player_x,player_o', [('viergewinnt', 'random', 'random'),
                                                    ('tictactoe', 'alpha', 'random')])
def test_play_smoketest(game, player_x, player_o):
    command = ['bin/play', '--game={}'.format(game), '-x', player_x, '-o', player_o, '--mcts-steps=3']
    assert run(command).returncode == 0
