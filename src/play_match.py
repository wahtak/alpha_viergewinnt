import click

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.viergewinnt import Viergewinnt, WinCondition, DrawCondition
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.human_player import HumanPlayer
from alpha_viergewinnt.player.mcts_player import MCTSPlayer, create_random_choice_strategy
from alpha_viergewinnt.match import play_match


def create_mcts_player(win_condition, loss_condition, draw_condition):
    return MCTSPlayer(
        win_condition=win_condition,
        loss_condition=loss_condition,
        draw_condition=draw_condition,
        selection_strategy=create_random_choice_strategy(),
        expansion_strategy=create_random_choice_strategy(),
        simulation_strategy=create_random_choice_strategy(),
        iterations=100,
        rollouts=100)


PLAYER_FACTORIES = {'random': RandomPlayer, 'human': HumanPlayer, 'mcts': create_mcts_player}


@click.command()
@click.option('-x', required=True, type=click.Choice(PLAYER_FACTORIES.keys()), help='Strategy for player X')
@click.option('-o', required=True, type=click.Choice(PLAYER_FACTORIES.keys()), help='Strategy for player O')
def cmd(x, o):
    """Play a Viergewinnt match"""
    game = Viergewinnt()
    player_x_win_condition = WinCondition(Player.X)
    player_o_win_condition = WinCondition(Player.O)
    draw_condition = DrawCondition()
    player_x = PLAYER_FACTORIES[x](
        win_condition=player_x_win_condition, loss_condition=player_o_win_condition, draw_condition=draw_condition)
    player_o = PLAYER_FACTORIES[o](
        win_condition=player_o_win_condition, loss_condition=player_x_win_condition, draw_condition=draw_condition)

    play_match(
        game=game,
        players={Player.X: player_x,
                 Player.O: player_o},
        win_conditions={Player.X: player_x_win_condition,
                        Player.O: player_o_win_condition},
        draw_condition=draw_condition,
        print_state=True,
        print_move=True,
        print_result=True)


if __name__ == '__main__':
    cmd()
