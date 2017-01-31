'''
Created on Mar 13, 2016

TODO: Add some description here.

@author: trucvietle
'''

from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
from textwrap import dedent
from itertools import *
from progressbar import ProgressBar
import numpy as np
import random
import math
import sys

def parse():
    '''
    Parses argumnents from the terminal.
    '''
    
    class CustomFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
        pass
    parser = ArgumentParser(formatter_class=CustomFormatter,
                            description=dedent('To be filled.'),
                            epilog=dedent('To be filled.'))
    parser.add_argument('-r', '--nrow', type=int, default=3, help='Number of rows')
    parser.add_argument('-c', '--ncol', type=int, default=4, help='Number of columns')
    parser.add_argument('-g', '--goal', type=int, default=1, help='Goal reward')
    parser.add_argument('-p', '--penalty', type=int, default=-1, help='Penalty')
    parser.add_argument('-d', '--discount', type=float, default=0.95, help='Discount')
    
    return vars(parser.parse_args())
    
def build_environment(nrow=3, ncol=4, goal_reward=1, penalty=-1):
    '''
    Sets up the MDP environment.
    '''

    ## Set the initial rewards
    global rewards
    rewards = np.zeros(shape=(nrow, ncol))
    
    ## The finite set of states are the grid cells
    global states 
    states = product(range(nrow), range(ncol))
    states = list(states) # convert iterator to list
    
    ## The goal state
    goal_row = 0
    goal_col = max(states)[1]
    global goal_state
    goal_state = (goal_row, goal_col)
    rewards[goal_state] = goal_reward
    
    ## The trap state
    trap_row = 1
    trap_col = max(states)[1]
    global trap_state
    trap_state = (trap_row, trap_col)
    rewards[trap_state] = penalty
    
    ## Coordinates of the 'wall'
    wall_row = int(nrow / 2)
    wall_col = int(ncol / 2) - 1
    global wall_state
    wall_state = (wall_row, wall_col)
    ## Set the reward values
    rewards[wall_state] = None
    
    ## Set the initial values
    global values 
    values = np.zeros(shape=(nrow, ncol))
    values[wall_state] = None
    values[goal_state] = goal_reward
    values[trap_state] = penalty
    
    ## The set of actions
    global actions
    actions = ['N', 'S', 'E', 'W']
    
    ## Create the transition probabilities
    ## Actions = north, south, east, west
    global transition 
    transition = {}
    transition['N'] = [0.8, 0, 0.1, 0.1] # north, south, east, west
    transition['S'] = [0, 0.8, 0.1, 0.1]
    transition['E'] = [0.1, 0.1, 0.8, 0]
    transition['W'] = [0.1, 0.1, 0, 0.8]
    
    ## The value of an action on the grid
    global action_values 
    action_values = {}
    action_values['N'] = [-1, 0] # x, y
    action_values['S'] = [1, 0]
    action_values['E'] = [0, 1]
    action_values['W'] = [0, -1]
        
def act(cur_state, action):
    '''
    Moves the agent through the states based on action taken.
    '''
    
    if cur_state == wall_state:
        sys.exit('Cannot enter wall state!')
    
    action_value = action_values[action]
    next_state = cur_state
    next_state = list(next_state) # convert from tuple to list
    
    ## Check if either goal or trap state has been reached
    if (cur_state == goal_state) or (cur_state == trap_state):
        ## Reached the goal or the trap - no change in state
        return cur_state
    ## Otherwise, update the state
    next_row = cur_state[0] + action_value[0]
    next_col = cur_state[1] + action_value[1]
    
    ## Constrained by the edge of the grid
    next_state[0] = min(max(states)[0], max(0, next_row))
    next_state[1] = min(max(states)[1], max(0, next_col))
    next_state = tuple(next_state) # convert from list to tuple
    if next_state == wall_state: # hit the wall - no change in state
        next_state = cur_state
    return next_state

def bellman_update(state, action, gamma=1):
    ## action = {'N', 'S', 'E', 'W'}
    ## state: the current state
    ## gamma: discount factor
    trans_prob = transition[action]
    q = [0] * len(trans_prob) # the Q-function
    for i in range(len(trans_prob)):
        if trans_prob[i] > 0:
            prob_action = actions[i]
            next_state = act(state, prob_action)
            q[i] = trans_prob[i] * (rewards[state] + gamma * values[next_state])
    return sum(q)

def value_iteration(nrow=3, ncol=4, goal_reward=1, penalty=-1,
                    gamma=1, n_iter=1000):
    ## Values of the previous iteration
    prev_values = np.zeros(shape=(nrow, ncol))
    prev_values[goal_state] = goal_reward
    prev_values[trap_state] = penalty
    prev_values[wall_state] = None
    ## The threshold of different
    epsilon = 1e-5
    ## Define a progress bar
    progress = ProgressBar(maxval=n_iter).start()
    ## Create a policy matrix
    policy = np.chararray(shape=(nrow, ncol))
    policy[goal_state] = '_'
    policy[trap_state] = '_'
    policy[wall_state] = '_'
    
    for i in range(n_iter):
        for state in states:
            if state != goal_state and state != trap_state and state != wall_state:
                q_values = [0] * len(actions)
                counter = 0
                for action in actions:
                    q_values[counter] = bellman_update(state, action, gamma)
                    counter += 1
                max_q = max(q_values)
                if math.isnan(max_q):
                    sys.exit('max_q is nan!')
                max_idx = q_values.index(max_q)
                values[state] = max_q
                policy[state] = actions[max_idx]
                
        diff = values - prev_values
        max_diff = np.nanmax(abs(diff))
        if max_diff < epsilon:
            progress.finish()
            break
        prev_values[:] = values # make a shallow copy of the list so that it won't get updated
        progress.update(i+1)
    return policy

def get_transition_matrix(policy):
    '''
    Computes an (|S|x|S|) transition matrix for the give policy.
    '''
    transition_matrix = np.zeros(shape=(len(states), len(states)))
    for state in states:
        ## The probabilities of transition to the next states given the current state and action
        if state != wall_state:
            from_state_idx = states.index(state)
            if state == goal_state or state == trap_state:
                ## Always stay in the same state no matter what action
                next_state_idx = from_state_idx
                transition_matrix[from_state_idx, next_state_idx] = 1
            else:
                action = policy[state]
                trans_prob = transition[action]
                for i in range(len(trans_prob)):
                    if trans_prob[i] > 0:
                        prob_action = actions[i]
                        next_state = act(state, prob_action)
                        next_state_idx = states.index(next_state)
                        if next_state == state: # stay the same
                            prev_value = transition_matrix[from_state_idx, next_state_idx]
                            transition_matrix[from_state_idx, next_state_idx] = prev_value + trans_prob[i]
                        else:
                            transition_matrix[from_state_idx, next_state_idx] = trans_prob[i]
    
    return transition_matrix

def policy_iteration(nrow=3, ncol=4, gamma=1, n_iter=1000):
    '''
    Implements the policy iteration algorithm.
    '''
    ## Initialize a random policy
    policy = np.chararray(shape=(nrow, ncol))
    policy[goal_state] = '_'
    policy[trap_state] = '_'
    policy[wall_state] = '_'
    for state in states:
        if state != goal_state and state != trap_state and state != wall_state:
            rand_action = random.choice(actions)
            policy[state] = rand_action
#     print policy
    
    ## Define a progress bar
    progress = ProgressBar(maxval=n_iter).start()
    ## Run the policy iteration loop
    for i in range(n_iter):
        ## Compute the value of the current policy
        ## Solve the linear equations
        transition_matrix = get_transition_matrix(policy)
#         print transition_matrix
        b = np.zeros(shape=(1, len(states))) # the RHS - rewards
        for state in states:
            state_idx = states.index(state)
            if state != wall_state:
                b[0, state_idx] = rewards[state]
        b = np.dot(b, transition_matrix)
        b = b.transpose()
        
        ## The matrix of coefficients
        a = np.identity(len(states)) - gamma * transition_matrix
        ## Solve for the values
        value = np.linalg.solve(a, b)
        ## Update the global value function
        for state in states:
            if state != wall_state:
                state_idx = states.index(state)
                values[state] = value[state_idx]
#         print values
        
        is_changed = False
        for state in states:
            if state != goal_state and state != trap_state and state != wall_state:
                q_best = values[state]
                for action in actions:
                    q_sa = bellman_update(state, action, gamma)
                    if q_sa > q_best and action != policy[state]:
                        policy[state] = action
                        q_best = q_sa
                        is_changed = True
        if not is_changed:
            progress.finish()
            break
        else:
#             print policy
            progress.update(i+1)
    return policy

if __name__ == '__main__':
    params = parse()
    nrow = params['nrow']
    ncol = params['ncol']
    goal_reward = params['goal']
    penalty = params['penalty']
    gamma = params['discount']
    
    build_environment(nrow=nrow, ncol=ncol, goal_reward=goal_reward, penalty=penalty)
#     print rewards
#     print values
#     print transition
    
#     policy = value_iteration(nrow, ncol, goal_reward=goal_reward, penalty=penalty,
#                              gamma=gamma, n_iter=1000)
    policy = policy_iteration(nrow, ncol, gamma=gamma, n_iter=1000)
    print policy
    print values
    