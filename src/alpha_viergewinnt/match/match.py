from ..game.board import Player
from ..game.viergewinnt import Viergewinnt, WinCondition, DrawCondition


def play_match(player_x, player_o, print_state, print_move, print_result):
    players = {Player.X: player_x, Player.O: player_o}
    win_conditions = {Player.X: WinCondition(Player.X), Player.O: WinCondition(Player.O)}
    draw_condition = DrawCondition()
    viergewinnt = Viergewinnt()

    while not viergewinnt.check(draw_condition):
        if print_state:
            print(viergewinnt)
        next_move = players[viergewinnt.current_player].get_next_move(viergewinnt)
        if print_move:
            print('Player %s plays %d' % (viergewinnt.current_player.name, next_move))
        viergewinnt.play_move(player=viergewinnt.current_player, move=next_move)

        if viergewinnt.check(win_conditions[viergewinnt.current_player]):
            if print_result:
                print(viergewinnt)
                print('Player %s wins!' % viergewinnt.current_player.name)
            return viergewinnt.current_player
