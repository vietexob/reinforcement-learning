import interface as bbox
 
n_features = 36
n_actions = 4
max_time = -1


def calc_best_action_using_checkpoint():
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
		bbox.reset_level()
	else:
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