# AlphaViergewinnt
Reimplementation of DeepMind's AlphaGo Zero for the game Viergewinnt (WIP)

[![pipeline status](https://gitlab.com/alexander.metz/alpha-viergewinnt/badges/master/pipeline.svg)](https://gitlab.com/alexander.metz/alpha-viergewinnt/commits/master)

[![coverage report](https://gitlab.com/alexander.metz/alpha-viergewinnt/badges/master/coverage.svg)](https://gitlab.com/alexander.metz/alpha-viergewinnt/commits/master)

_WORK IN PROGRESS_

## Current state
  - [x] Viergewinnt game logic
  - [x] Basic Monte-Carlo tree search
  - [ ] Reinforcement learning model as MCTS strategy

## Usage
    $ python play_match.py --help
    Usage: play_match.py [OPTIONS]

      Play a match

    Options:
      --game [tictactoe|viergewinnt]  Game to be played  [required]
      -x [random|human|mcts]          Strategy for player X  [required]
      -o [random|human|mcts]          Strategy for player O  [required]
      --help                          Show this message and exit.

    $ python play_match.py -x human -o mcts --game viergewinnt

## MCTS visualization
Visualization of MCTS search tree with networkx

![mcts](/docs/tree_search.png?raw=true)
