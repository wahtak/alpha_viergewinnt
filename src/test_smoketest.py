from random import Random

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.viergewinnt import Viergewinnt, WinCondition, DrawCondition
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.match import play_match


def test_smoketest():
    # initialize random players with seeded random number generator for deterministic test
    players = {Player.X: RandomPlayer(Random(0)), Player.O: RandomPlayer(Random(1))}
    win_conditions = {Player.X: WinCondition(Player.X), Player.O: WinCondition(Player.O)}
    draw_condition = DrawCondition()
    game = Viergewinnt()

    winner = play_match(
        game=game,
        players=players,
        win_conditions=win_conditions,
        draw_condition=draw_condition,
        print_state=False,
        print_move=False,
        print_result=False)

    assert isinstance(winner, Player)
