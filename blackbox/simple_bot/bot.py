## Import the game simulator
import interface as bbox
import pandas as pd
import random
import calendar
import time

def get_action_by_state(state, verbose=False):
    '''
    This is the policy function. It takes the environment state vector and returns an
    action that the agent performs. It suffices to only modify this function to create
    a proper learning agent.
    '''
    
    an_interaction = []
    if verbose: # enables detailed logging
        for i in range(n_features):
            ## Print the environment state vector
#             print ("state[%d] = %f" % (i, state[i]))
            an_interaction.append(state[i])
        ## Print the current score and time (number of current game steps)
        reward = bbox.get_score()
        an_interaction.append(reward)
#         print ("score = {}, time = {}".format(reward, bbox.get_time()))
        
    ## TODO: Change this action
    action_to_do = random.random()
    an_interaction.append(action_to_do)
    interaction_list.append(an_interaction)
    return action_to_do

## Need **not** modify the code below, but it is useful to understand what it does
n_features = n_actions = max_time = -1

def prepare_bbox():
    global n_features, n_actions, max_time
    ## TODO: Save the interactions with the environment as an output data frame
    global interaction_list
    interaction_list = []
    
    ## Reset the environment to initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        ## Load the game level
        bbox.load_level("../levels/train_level.data", verbose=True)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()
        
        ## The matrix that contains the output data frame
        states = ['state_'] * n_features
        state_list = [states[i] + str(i) for i in range(n_features)]
        header_list = state_list + ['reward', 'action']
        interaction_list.append(header_list)
        
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
        action = get_action_by_state(state, verbose=verbose)
        ## Function do_action(action) returns False if the level
        ## is finished; otherwise, it returns True
        has_next = bbox.do_action(action)
    
    ## Save the interactions as an output CSV file
    headers = interaction_list.pop(0)
    interaction_df = pd.DataFrame(interaction_list, columns=headers)
    datetime_int = int(calendar.timegm(time.gmtime()))
    out_filename = '../output/interaction_' + str(datetime_int) + '.csv'
    interaction_df.to_csv(out_filename, index=False)
    print 'Saved to file: ' + out_filename
    
    ## When submitting solution, make sure to call finish(), which returns the sum of points obtained
    ## during the entire simulation. This number is used as the public leader board score
    bbox.finish(verbose=True)

if __name__ == "__main__":
    run_bbox(verbose=True)
