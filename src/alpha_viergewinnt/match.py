def play_match(game, players, win_conditions, draw_condition, print_state, print_move, print_result):
    while not game.check(draw_condition):
        if print_state:
            print(game)
        next_move = players[game.current_player].get_next_move(game)
        if print_move:
            print('Player %s plays %d' % (game.current_player.name, next_move))

        last_player = game.current_player
        game.play_move(player=game.current_player, move=next_move)

        if game.check(win_conditions[last_player]):
            if print_result:
                print(game)
                print('Player %s wins!' % last_player.name)
            return game.current_player
