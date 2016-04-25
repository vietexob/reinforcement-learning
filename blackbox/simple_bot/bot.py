## Import the game simulator
import interface as bbox

def get_action_by_state(state, verbose=0):
    '''
    This is the main function, the agent's brain. It takes the env state vector and returns an
    action that the agent is to perform. It suffices to only modify this function to create a proper agent.
    '''
    if verbose: # enables detailed logging
        for i in range(n_features):
            ## Print the env state vector
            print ("state[%d] = %f" %  (i, state[i]))
        ## Print the current score and time (no. of current game steps)
        print ("score = {}, time = {}".format(bbox.get_score(), bbox.get_time()))

    ## Always performs action '0' -- not so smart!
    action_to_do = 0
    return action_to_do

## Participants need not modify the code below, but it could be useful to understand what it does
n_features = n_actions = max_time = -1
 
def prepare_bbox():
    global n_features, n_actions, max_time
    
    ## Reset the env to initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        ## Load the game level
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
 
 
def run_bbox(verbose=False):
    has_next = 1
    
    ## Prepare the env -- load the game level
    prepare_bbox()
    
    while has_next:
        ## Get the current env state
        state = bbox.get_state()
        ## Choose an action to perform at the current step
        action = get_action_by_state(state)
        ## Perform the chosen action
        ## Function do_action(action) returns False if level is finished; otherwise, returns True
        has_next = bbox.do_action(action)
    
    ## Finish the game simulation, print the earned reward
    ## When submitting solution, make sure to call finish(), which returns the sum of pts obtained
    ## This number is used as the public leaderboard score
    bbox.finish(verbose=1)
 
if __name__ == "__main__":
    run_bbox(verbose=0)
 