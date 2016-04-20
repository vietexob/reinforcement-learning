# Train a Smartcab How to Drive

## Synopsis
Smartcab is a self-driving car from the not-so-distant future that ferries people from one arbitrary location to another. This application demonstrates how to use model-free **reinforcement learning** (i.e., Q-learning) to train smartcabs to desired self-driving behaviors through trial and error.

Refer to [these slides](https://docs.google.com/presentation/d/1fJPmHzDFc9SYykhgZIRH-ZovdRat6cHJOBNG4CzTA60/edit?usp=sharing) for more info.

Video of my talk at [PUGS](http://pugs.org.sg/) meetup on April 18, 2016: [https://engineers.sg/video/reinforcement-learning-using-python-python-sg--678](https://engineers.sg/video/reinforcement-learning-using-python-python-sg--678).

## Install

This project requires Python 2.7x with the [Pygame](https://www.pygame.org/wiki/GettingStarted) library installed.

Notes on installation of Pygame:
- If using `home-brew`: Remember to update `home-brew` to the latest version before installing or things could go haywire.
- Important to install all dependencies beforehand!
- Using `conda`: `conda install -c https://conda.anaconda.org/quasiben pygame`

## Run

Make sure you are in the lowest-level project directory `smartcab/smartcab` (that contains `agent.py`). Then run:

```python agent.py```

## Parameters


## Demos

In these demonstrations, the following params are fixed: `trials=1`, `delay=0.50`, `alpha=0.20`, `gamma=1-4/deadline`, `epsilon=0.10`, `initial=0`, and `deadline=True`. The following scenarios are demonstrated:
- Learning with no history: `history=0`, `delay=1.0`
- Learning through trial and error: `trials=10`
- Learning with history: `history=50`

## Parameter Tuning


## Directories

