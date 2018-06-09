from copy import deepcopy


def play_match(game, players, win_conditions, draw_condition, print_state, print_move, print_result):
    while not game.check(draw_condition):
        if print_state:
            print(game)
        next_move = players[game.current_player].get_next_move(game)
        if print_move:
            print('Player %s plays %s' % (game.current_player.name, next_move))

        last_player = game.current_player
        game.play_move(player=game.current_player, move=next_move)

        if game.check(win_conditions[last_player]):
            if print_result:
                print(game)
                print('Player %s wins!' % last_player.name)
            return last_player

    if print_result:
        print(game)
        print('Draw!')
    return None


def evaluate_players(iterations, game, players, win_conditions, draw_condition):
    results = {player: 0 for player in players}
    results[None] = 0

    for iteration in range(iterations):
        winner = play_match(
            deepcopy(game),
            players,
            win_conditions,
            draw_condition,
            print_state=False,
            print_move=False,
            print_result=False)
        results[winner] += 1

    return results
