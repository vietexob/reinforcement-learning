# Train a Smartcab How to Drive

## Synopsis
Smartcab is a self-driving car from the not-so-distant future that ferries people from one arbitrary location to another. This application demonstrates how to use the model-free **reinforcement learning** (i.e., Q-learning) to train smartcab to desired self-driving behaviors through trial and error.

* Refer to [these slides](https://docs.google.com/presentation/d/1fJPmHzDFc9SYykhgZIRH-ZovdRat6cHJOBNG4CzTA60/edit?usp=sharing) for more detailed info.
* Video of my talk at [PUGS](http://pugs.org.sg/) meetup on April 18, 2016: [https://engineers.sg/video/reinforcement-learning-using-python-python-sg--678](https://engineers.sg/video/reinforcement-learning-using-python-python-sg--678).

## Installation

This project requires Python 2.7x with the [Pygame](https://www.pygame.org/wiki/GettingStarted) library installed.

Notes on installation of Pygame:
- If using `home-brew`, make sure to update it to the latest version before installing or things could go haywire.
- It is important to install all dependencies beforehand!
- If using [anaconda](https://www.continuum.io/downloads): `conda install -c https://conda.anaconda.org/quasiben pygame`; or search the conda repo: `anaconda search -t conda pygame`.
- [This tutorial](http://kidscancode.org/blog/2015/09/pygame_install/) is very helpful.

[Pandas](http://pandas.pydata.org/) library is also required -- make you have it properly installed.

## Run

Make sure you are in the **lowest-level** project directory `./smartcab/smartcab` (that contains `agent.py`). Then run:

```python agent.py```

There are **options** to be passed as follows:
* `--trials`: the number of trials to be run (each trial is a new "game" with new deadline and destination). Q-function is accumulated and averaged over the trials so that later trials are expected to perform better than initial ones. Default is `trials=100`.
* `--delay`: to control the simulation speed, lower delay for higher speed. Default is `delay=0.50`.
* `--log`: the output log file to be saved. Default is `../smartcab.log`.
* `--dummies`: the number of dummy agents that cause traffic on the grid network. Default is `dummies=3`.
* `--alpha`: the learning rate in `[0..1]`. Lower alpha, less learning, higher alpha, more learning. Default is `alpha=0.20`.
* `--gamma`: the discount factor in `[0..1]` of future reward. Lower gamma, more discount, higher gamma, less discount. Default is `gamma=0.90`.
* `--epsilon`: the probability of random action selection. Lower epsilon, less random (more greedy), higher epsilon, more random. Default is `epsilon=0.10`.
* `--initial`: the initial value of Q-function for every <state, action> tuple. Default is `initial=0`.
* `--deadline`: whether to set `gamma = 1 - c/deadline`, where `c` is some positive constant and `deadline` is the given deadline of the trial. Default is `deadline=True`, which effectively overrides the set value of `gamma`. 
* `--history`: the number of previous successful trials to consider for initialization of Q-function. Default is `history=0`. Setting `history` effectively overrides the set value of `initial`.

## Demos

The following three (3) scenarios are considered for demonstrations with fixed parameters (unless explicitly specified): `alpha=0.20`, `gamma=1-4/deadline`, `epsilon=0.10`, `initial=0`, and `deadline=True`:
1. Learning with no history: `trials=1`, `history=0`, `delay=1.0`. Agent is expected to behave erratically with low chance of successfully reaching the destination in time. 
2. Learning over the trials: `trials=10`, `history=0`, `delay=0.25`. Agent is expected to behave erratically at first, but then gradually improves performance as it learns over the trials. The more trials, the smarter it gets.
3. Learning with history: `trials=1`, `history=50`, `delay=0.50`. Agent is expected to behave reasonably and successfully reach the destination in time. Agent gets smarter as more histories are considered to initialize the Q-function.

## Parameter Tuning

Infinitely large number of combinations of learning parameters `alpha`, `gamma`, `epsilon` and `initial` are possible for Q-learning. This is where the black magic lies! 

## Directories

The directory `./q_tables` stores the learned Q-functions (as CSV files without headers) of the previous "successful" runs. A run is *successful* if it contains at least 10 trials and in the *last* 10 trials, the agent successfully reached destination within deadline in at least 7 of them.
