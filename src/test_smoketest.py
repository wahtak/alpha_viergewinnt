import pytest
from random import Random

from alpha_viergewinnt.game.board import Player
from alpha_viergewinnt.game.viergewinnt import Viergewinnt, WinCondition, DrawCondition
from alpha_viergewinnt.player.random_player import RandomPlayer
from alpha_viergewinnt.player.mcts_player import MCTSPlayer, create_random_choice_strategy
from alpha_viergewinnt.match import play_match, evaluate_players


@pytest.fixture
def default_setup():
    # initialize random players with seeded random number generator for deterministic test
    random = Random(1)
    game = Viergewinnt()
    player_x_win_condition = WinCondition(Player.X)
    player_o_win_condition = WinCondition(Player.O)
    win_conditions = {Player.X: player_x_win_condition, Player.O: player_o_win_condition}
    draw_condition = DrawCondition()
    mcts_player_x = MCTSPlayer(
        win_condition=player_x_win_condition,
        loss_condition=player_o_win_condition,
        draw_condition=draw_condition,
        selection_strategy=create_random_choice_strategy(random),
        expansion_strategy=create_random_choice_strategy(random),
        simulation_strategy=create_random_choice_strategy(random),
        iterations=3,
        rollouts=3)
    random_player_o = RandomPlayer(random)
    players = {Player.X: mcts_player_x, Player.O: random_player_o}
    return game, players, win_conditions, draw_condition


def test_smoketest_single_match(default_setup):
    game, players, win_conditions, draw_condition = default_setup
    winner = play_match(
        game=game,
        players=players,
        win_conditions=win_conditions,
        draw_condition=draw_condition,
        print_state=True,
        print_move=True,
        print_result=True)
    assert isinstance(winner, Player) or winner is None


def test_smoketest_evaluation(default_setup):
    game, players, win_conditions, draw_condition = default_setup
    iterations = 3
    results = evaluate_players(
        iterations=iterations, game=game, players=players, win_conditions=win_conditions, draw_condition=draw_condition)
    assert sum(results.values()) == iterations
