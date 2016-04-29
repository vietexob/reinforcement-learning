'''
Created on Apr 28, 2016

Implementation of vanilla Q-learning to the Blackbox challenge.

@author: trucvietle
'''

import interface as bbox
import random
import math

def sigmoid(x):
    '''
    The logistic function that maps any real number x to the range [0, 1]
    '''
    
    return 1 / (1 + math.exp(-x))

def get_state_tuple(state):
    '''
    Returns the discretized state vector.
    '''
    
    state_vector = []
    for i in range(n_features):
        sigmoid_state = sigmoid(state[i])
        state_var = int(sigmoid_state * 10)
        state_vector.append(state_var)
        
    return tuple(state_vector)

def get_action(state_tuple, verbose=False, is_current=True):
    '''
    Implementation of an action selection algorithm.
    The four possible actions to be returned are: {0, 1, 2, 3}.
    '''
    
    action = 0
#     if verbose:
#         for i in range(n_features):        
#             print ("state[%d] = %f" % (i, state[i]))
#         reward = bbox.get_score()
#         print reward
    if state_tuple in q_function:
        ## Find the action that yields the highest value
        action_function = q_function[state_tuple]
        greedy_action = max(action_function, key = action_function.get)
        if is_current:
            ## Generate a random action
            rand_action = random.choice(valid_actions)
            ## Select action using epsilon-greedy heuristic
            rand_num = random.random()
            action = greedy_action if rand_num <= (1 - epsilon) else rand_action
        else:
            action = greedy_action
    else:
        ## Initialize <state, action> pairs and select random action
        action_function = {}
        for action in valid_actions:
            action_function[action] = init_value
        q_function[state_tuple] = action_function
        action = random.choice(valid_actions)
        
    return action

## Set up the environment
n_features = 0
n_actions = 0
max_time = 0
q_function = {} # the Q-table
epsilon = 0.10 # probability of random action
gamma = 0.95 # discount factor
alpha = 0.20 # learning rate
valid_actions = [0, 1, 2, 3]
init_value = 0

def prepare_bbox():
    '''
    Prepares the environment (learning/test data).
    '''
    
    global n_features
    global n_actions
    global max_time
    global q_function
    global epsilon
    global gamma
    global alpha
    global valid_actions
    global init_value
    
    if bbox.is_level_loaded():
        ## Reset the environment to initial state
        bbox.reset_level()
    else:
        ## Load the training/test data
        bbox.load_level('../levels/train_level.data', verbose=True)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()

def run_bbox(verbose=False):
    '''
    Runs the Blackbox challenge.
    '''
    
    has_next = True
    prepare_bbox()
    
    while has_next:
        ## Observe the current state variables
        state = bbox.get_state()
        state_tuple = get_state_tuple(state)
        ## Select the current action
        action = get_action(state_tuple, verbose=verbose, is_current=True)
        ## Get the current reward
        reward = bbox.get_score()
        print 'Reward = ' + str(reward)
        
        ## Retrieve the current Q-value
        current_q = q_function[state_tuple][action]
        print 'Current Q = ' + str(current_q)
        
        ## Observe the next state (assuming there always is)
        has_next = bbox.do_action(action)
        next_state = bbox.get_state()
        next_state_tuple = get_state_tuple(next_state)
        ## Get the best q_action in the new state
        next_action = get_action(next_state_tuple, verbose=verbose, is_current=False)    
        ## Get the new Q_value
        next_q = q_function[next_state_tuple][next_action]
        ## Update the Q-function
        q_function[state_tuple][action] = (1 - alpha) * current_q + alpha * (reward + gamma * next_q)
        print 'Updated Q = ' + str(q_function[state_tuple][action])
    
    bbox.finish(verbose=True)
    
if __name__ == '__main__':
    run_bbox(verbose=True)
    