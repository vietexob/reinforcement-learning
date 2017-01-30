# MDP: Robot Navigating in a Grid World
This implements the canonical example of a Markov decision process (MDP): robot navigating in a grid world with perfect information.

## Synopsis:

## Run

Run from the terminal:

`python grid_mdp.py`

The following options are available as arguments:
- `--nrow`: the number of rows of the grid world. Default: `nrow = 3`
- `--ncol`: the number of columns of the grid world. Default: `ncol = 4`
- `--goal`: the *positive integer* representing the *reward*. Default: `goal = 1`
- `--penalty`: the *negative integer* representing the *penalty*. Default: `penalty = -1`
- `--discount`: the discount factor (float) for infinite-horizon MDP. Default: `discount = 0.95`
