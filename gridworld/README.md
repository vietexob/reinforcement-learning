# MDP: Robot Navigating in a Grid World

## Synopsis:
This program implements the *canonical* example of a Markov decision process (MDP): a robot navigating in a grid world with perfect information. Click [here](http://artint.info/html/ArtInt_224.html#gridworld-fig) for detailed description of the example. In short, a "grid world" is a rectangular grid environment defined by its length (`ncol`) and width (`nrow`) dimensions. Each grid cell represents a *state*. The "goal" state is located at the top right corner of the grid. The "trap" state is located directly horizontally below the goal state. The "wall" is located in the "center" of the grid, i.e., at coordinates: `nrow/2` and `ncol/2` (rounded down, if any). The gaol state is initialized with a positive reward, while the trap state with a negative penalty. The wall has `nan` reward, and the rest of the states are initialized to `0` reward. The robot moves one step at a time, in either north, south, east, or west direction. Its movements are, however, restricted by the boundaries of the grid and the wall: it cannot get into the wall state. The robot's goal is to end up at the goal state and gets rewarded. It also wants to avoid the trap state, since it gets penalized there. In either state (goal or trap), the game terminates.

The robot is initially located at the bottom right corner of the grid. At each time step, it makes a move according to a "learned" policy. Each move changes its location: with probability 0.80, it moves according to the intended direction, and probability 0.10 it moves in *either* sideways direction (i.e., it cannot move in the opposite direction). The purpose of MDP is to figure an optimal policy for the robot so that it can reach the goal state in the least number of moves. This is done through the principle of reinforcement learning: trial and error. This program implements two such algorithms: value iteration and policy iteration that can be passed as argument. It also supports discounted reward (`gamma`) for faster convergence.

## Dependencies:
The following packages are required: `numpy` and `progressbar`. Installing [Anaconda Python](https://www.continuum.io/downloads) is highly recommended to preempt most dependency problems.

## Run

Run from the terminal:

`python grid_mdp.py`

The following options are available as arguments:
- `--nrow`: the number of rows of the grid world. Default: `nrow = 3`
- `--ncol`: the number of columns of the grid world. Default: `ncol = 4`
- `--goal`: the *positive integer* representing the *reward*. Default: `goal = 1`
- `--penalty`: the *negative integer* representing the *penalty*. Default: `penalty = -1`
- `--discount`: the discount factor (float) `gamma` for infinite-horizon MDP. Default: `discount = 0.95`
- `--value`: whether to run value iteration (`value = 1`) or policy iteration (otherwise). Default: `value = 1`
