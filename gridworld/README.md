# MDP: Robot Navigating in a Grid World

## Synopsis:
This program implements the *canonical* example of a Markov decision process (MDP): robot navigating in a grid world with perfect information. Click [here](http://artint.info/html/ArtInt_224.html#gridworld-fig) for detailed description of the example. In short, a "grid world" is a rectangular grid environment defined by its length (`ncol`) and width (`nrow`). Each grid cell represents a *state*. The "goal" state is located at the top right corner of the grid. The "trap" state is located directly below the goal state horizontally. The "wall" is located in the "center" of the grid, i.e., at the coordinates `nrow/2` and `ncol/2`. The gaol state is initialized with a positive reward, while the admits a negative penalty. The wall has `nan` reward, and the rest of the states are initialized to `0` reward. The robot can move one step at a time, in either north, south, east, or west direction. Its movements are, however, restricted by the boundaries of the grid as well as the wall: it cannot get into the wall state. The robot's goal is to end up at the goal state and gets rewarded. It also wants to avoid the trap state, since it gets penalized there. In either state (goal or trap). The game terminates. 

The robot is initially located at the bottom right corner of the grid. At each time step, it makes a move according to a "learned" policy. Each move changes its location according to the move. However, The purpose of MDP is to 

## Dependencies:
The following packages are required: `numpy` and `progressbar`. Installing [Anaconda Python](https://www.continuum.io/downloads) is highly recommended to resolve dependency problems.

## Run

Run from the terminal:

`python grid_mdp.py`

The following options are available as arguments:
- `--nrow`: the number of rows of the grid world. Default: `nrow = 3`
- `--ncol`: the number of columns of the grid world. Default: `ncol = 4`
- `--goal`: the *positive integer* representing the *reward*. Default: `goal = 1`
- `--penalty`: the *negative integer* representing the *penalty*. Default: `penalty = -1`
- `--discount`: the discount factor (float) `gamma` for infinite-horizon MDP. Default: `discount = 0.95`
