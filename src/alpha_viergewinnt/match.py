def play_match(game, players, win_conditions, draw_condition, print_state, print_move, print_result):
    while not game.check(draw_condition):
        if print_state:
            print(game)
        next_move = players[game.current_player].get_next_move(game)
        if print_move:
            print('Player %s plays %d' % (game.current_player.name, next_move))
        game.play_move(player=game.current_player, move=next_move)

        if game.check(win_conditions[game.current_player]):
            if print_result:
                print(game)
                print('Player %s wins!' % game.current_player.name)
            return game.current_player
