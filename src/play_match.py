import click

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.viergewinnt import Viergewinnt, WinCondition, DrawCondition
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.human_player import HumanPlayer
from alpha_viergewinnt.match import play_match

PLAYERS = {'random': RandomPlayer, 'human': HumanPlayer}


@click.command()
@click.option('-x', required=True, type=click.Choice(PLAYERS.keys()), help='Algorithm for player X')
@click.option('-o', required=True, type=click.Choice(PLAYERS.keys()), help='Algorithm for player O')
def cmd(x, o):
    """Play a Viergewinnt match"""
    players = {Player.X: PLAYERS[x](), Player.O: PLAYERS[o]()}
    win_conditions = {Player.X: WinCondition(Player.X), Player.O: WinCondition(Player.O)}
    draw_condition = DrawCondition()
    game = Viergewinnt()

    play_match(
        game=game,
        players=players,
        win_conditions=win_conditions,
        draw_condition=draw_condition,
        print_state=True,
        print_move=True,
        print_result=True)


if __name__ == '__main__':
    cmd()
