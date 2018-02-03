from random import Random

from ..game.board import Player
from ..player.random_player import RandomPlayer
from .match import play_match


def test_smoketest_match():
    # initialize random players with seeded random number generator for deterministic test
    player_x = RandomPlayer(Random(0))
    player_o = RandomPlayer(Random(1))
    winner = play_match(player_x=player_x, player_o=player_o, print_state=False, print_move=False, print_result=False)

    assert isinstance(winner, Player)
