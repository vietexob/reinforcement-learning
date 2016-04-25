'''
The main idea of checkpoints is simple: You can save the game and then load.
The agent chooses an action, repeats it in the next 100 time steps and sees what happens.
The agent repeats it for all 4 possible actions, and then actually goes by the trace that
leads him to the better score. This agent gets 37,815 points on the training level,
that's almost **20 times** higher that the baseline agent! Obviously this approach does
not have much predictive power and the agent is significantly over-fitted.

NOTE: Checkpoints are prohibited in the test mode.
You may not submit solutions using checkpoints to the server.
For documentation of checkpoint-related functions: http://blackboxchallenge.com/specs/
'''

import interface as bbox

## TODO: How are these defined?
n_features = 36
n_actions = 4
max_time = -1

def calc_best_action_using_checkpoint():
	## Create a checkpoint and get its ID
	checkpoint_id = bbox.create_checkpoint()
	best_action = -1
	best_score = -1e9
	
	for action in range(n_actions):
		for _ in range(100):
			bbox.do_action(action)
		
		if bbox.get_score() > best_score:
			best_score = bbox.get_score()
			best_action = action

		bbox.load_from_checkpoint(checkpoint_id)

	return best_action

def prepare_bbox():
	global n_features, n_actions, max_time
	
	if bbox.is_level_loaded():
		## Calling reset_level leaves checkpoints unmodified
		bbox.reset_level()
	else:
		## Calling load_level erases all checkpoints
		bbox.load_level("../levels/train_level.data", verbose=1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
		max_time = bbox.get_max_time()

def run_bbox():
	has_next = 1
	prepare_bbox()
	
	while has_next:
		best_act = calc_best_action_using_checkpoint()	
		for _ in range(100):
			has_next = bbox.do_action(best_act)

		if bbox.get_time() % 10000 == 0:
			print ("time = %d, score = %f" % (bbox.get_time(), bbox.get_score()))

	bbox.finish(verbose=1)

if __name__ == "__main__":
	run_bbox()
	