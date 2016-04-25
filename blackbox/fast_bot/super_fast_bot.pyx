import interface as bbox
cimport interface as bbox


cdef int get_action_by_state_fast(float* state, int verbose=0):
    cdef:
        int i, action_to_do

    if verbose == 1:
        for i in xrange(n_features):
            print ("state[%d] = %f" %  (i, state[i]))

        print ("score = {}, time = {}".format(bbox.c_get_score(), bbox.c_get_time()))

    action_to_do = 0
    return action_to_do


cdef int n_features, n_actions
 
 
def prepare_bbox():
    global n_features, n_actions
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
 
def run_bbox():
    cdef:
        float* state
        int action, has_next = 1
    
    prepare_bbox()
 
    while has_next:
        state = bbox.c_get_state()
        action = get_action_by_state_fast(state)
        has_next = bbox.c_do_action(action)
 
    bbox.finish(verbose=1)