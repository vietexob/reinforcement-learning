'''
Created on Apr 28, 2016

Implementation of vanilla Q-learning to the Blackbox challenge.

@author: trucvietle
'''

import interface as bbox
import math

def sigmoid(x):
    '''
    The logistic function that maps any real number x to the range [0, 1]
    '''
    
    return 1 / (1 + math.exp(-x))

def get_action(state, verbose=False):
    '''
    Implementation of an action selection algorithm.
    The four possible actions to be returned are: {0, 1, 2, 3}.
    '''
    if verbose:
        for i in range(n_features):
            sigmoid_state = sigmoid(state[i])
            state_var = int(sigmoid_state * 10)
            print ("state[%d] = %f" % (i, state_var))
    
    reward = bbox.get_score()
    print reward
    action = 0
    return action

## Set up the environment
n_features = 0
n_actions = 0
max_time = 0

def prepare_bbox():
    '''
    Prepares the environment (learning/test data).
    '''
    
    global n_features
    global n_actions
    global max_time
    
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
        state = bbox.get_state()
        action = get_action(state, verbose=verbose)
        has_next = bbox.do_action(action)
    
    bbox.finish(verbose=True)
    
if __name__ == '__main__':
    run_bbox(verbose=True)
    