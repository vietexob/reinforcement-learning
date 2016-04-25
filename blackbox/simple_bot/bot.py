## Import the game simulator
import interface as bbox

def get_action_by_state(state, verbose=False):
    '''
    This is the policy function. It takes the environment state vector and returns an
    action that the agent performs. It suffices to only modify this function to create
    a proper learning agent.
    '''
    
    if verbose: # enables detailed logging
        for i in range(n_features):
            ## Print the environment state vector
            print ("state[%d] = %f" % (i, state[i]))
        ## Print the current score and time (number of current game steps)
        print ("score = {}, time = {}".format(bbox.get_score(), bbox.get_time()))
        
    ## TODO: Always performs action '0' -- not so smart!
    action_to_do = 0
    return action_to_do

## Need **not** modify the code below, but it is useful to understand what it does
n_features = n_actions = max_time = -1

def prepare_bbox():
    global n_features, n_actions, max_time
    
    ## Reset the environment to initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        ## Load the game level
        bbox.load_level("../levels/train_level.data", verbose=True)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
        
def run_bbox(verbose=False):
    '''
    Runs the Blackbox challenge.
    '''
    
    has_next = True
    
    ## Prepare the environment -- load the game level
    prepare_bbox()
    
    while has_next:
        ## Get the current environment state vector
        state = bbox.get_state()
        ## Choose an action to perform at the current state
        action = get_action_by_state(state)
        ## Function do_action(action) returns False if the level
        ## is finished; otherwise, it returns True
        has_next = bbox.do_action(action)
    
    ## Finish the game simulation, print the earned reward
    ## When submitting solution, make sure to call finish(), which returns the sum of points obtained
    ## This number is used as the public leader board score
    bbox.finish(verbose=True)

if __name__ == "__main__":
    run_bbox(verbose=True)
