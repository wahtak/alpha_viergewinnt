from random import Random

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.viergewinnt import Viergewinnt, WinCondition, DrawCondition
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.mcts_player import MCTSPlayer, create_random_choice_strategy
from alpha_viergewinnt.match import play_match


def test_smoketest():
    # initialize random players with seeded random number generator for deterministic test
    random = Random(1)
    game = Viergewinnt()
    player_x_win_condition = WinCondition(Player.X)
    player_o_win_condition = WinCondition(Player.O)
    draw_condition = DrawCondition()
    mcts_player_x = MCTSPlayer(
        win_condition=player_x_win_condition,
        loss_condition=player_o_win_condition,
        draw_condition=draw_condition,
        selection_strategy=create_random_choice_strategy(random),
        expansion_strategy=create_random_choice_strategy(random),
        simulation_strategy=create_random_choice_strategy(random),
        iterations=10,
        rollouts=10)
    random_player_o = RandomPlayer(random)

    winner = play_match(
        game=game,
        players={Player.X: mcts_player_x,
                 Player.O: random_player_o},
        win_conditions={Player.X: player_x_win_condition,
                        Player.O: player_o_win_condition},
        draw_condition=draw_condition,
        print_state=True,
        print_move=True,
        print_result=True)

    assert isinstance(winner, Player)


if __name__ == '__main__':
    test_smoketest()
