from alpha_viergewinnt.player.pure_mcts_player import PureMctsPlayer, create_random_choice_strategy


def create_pure_mcts_player(player, mcts_steps=30, mcts_rollouts=30, random_seed=None):
    return PureMctsPlayer(
        player=player,
        selection_strategy=create_random_choice_strategy(random_seed),
        expansion_strategy=create_random_choice_strategy(random_seed),
        simulation_strategy=create_random_choice_strategy(random_seed),
        mcts_steps=mcts_steps,
        mcts_rollouts=mcts_rollouts)
