import sys
import csv
import math
import time
import random
import calendar

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from progressbar import ProgressBar

class LearningAgent(Agent):
    """An agent that learns how to drive in the smartcab world."""
    
    def __init__(self, env, init_value=0, gamma=0.90, alpha=0.20, epsilon=0.50):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override default color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        ## Initialize the Q-function as a dictionary (state) of dictionaries (actions)
        self.q_function = {}
        ## Initial value of any (state, action) tuple is an arbitrary random number
        self.init_value = init_value
        ## Discount factor gamma: 0 (myopic) vs 1 (long-term optimal)
        self.gamma = gamma
        ## Learning rate alpha: 0 (no learning) vs 1 (consider only most recent information)
        ## NOTE: Normally, alpha decreases over time: for example, alpha = 1 / t
        self.alpha = alpha
        ## Parameter of the epsilon-greedy action selection strategy
        ## NOTE: Normally, epsilon should also be decayed by the number of trials
        self.epsilon = epsilon
        
        ## The trial number
        self.trial = 1
        ## The cumulative reward
        self.cumulative_reward = 0
        
    def get_q_function(self):
        return self.q_function
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.env.set_trial_number(self.trial)
        self.trial += 1
        
        ## Decay the epsilon parameter
        self.epsilon = self.epsilon / math.sqrt(self.trial)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.cumulative_reward = 0
    
    def select_action(self, state=None, is_current=True, t=1):
        '''
        Implements the epsilon-greedy action selection that selects the best-valued action in this state
        (if is_current) with probability (1 - epsilon) and a random action with probability epsilon.
        't' is the current time step that can be used to modify epsilon.
        '''
        if state in self.q_function:
            ## Find the action that has the highest value
            action_function = self.q_function[state]
            q_action = max(action_function, key = action_function.get)
            if is_current:
                ## Generate a random action
                rand_action = random.choice(self.env.valid_actions)
                ## Select action using epsilon-greedy heuristic
                rand_num = random.random()
                action = q_action if rand_num <= (1 - self.epsilon) else rand_action
            else:
                action = q_action
        else:
            ## Initialize <state, action> pairs and select random action
            action_function = {}
            for action in self.env.valid_actions:
                action_function[action] = self.init_value
            self.q_function[state] = action_function
            action = random.choice(self.env.valid_actions)        
        
        return action
        
    def update(self, t):
        '''
        At each time step t, the agent:
        - Is given the next waypoint (relative to its current location and direction)
        - Senses the intersection state (traffic light and presence of other vehicles)
        - Gets the current deadline value (time remaining)
        '''
        ## The destination trying to reach
#         destination = self.env.agent_states[self]['destination']
        
        ## Observe the current state variables
        ## (1) Traffic variables
        inputs = self.env.sense(self)
        light = inputs['light']
        oncoming = inputs['oncoming']
#         right = inputs['right']
        left = inputs['left']
        ## (2) Direction variables
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
#         deadline = self.env.get_deadline(self)
#         location = self.env.agent_states[self]['location']
#         distance = self.env.compute_dist(location, destination)
#         heading = self.env.agent_states[self]['heading']
        
        ## Update the current observed state
        self.state = (light, oncoming, left, self.next_waypoint)
        current_state = self.state # save this for future use
        
        ## Select the current action
        action = self.select_action(state=current_state, is_current=True, t=t)
        ## Execute action, get reward and new state
        reward = self.env.act(self, action)
        self.cumulative_reward += reward
        self.env.set_cumulative_reward(self.cumulative_reward)
        
        ## Retrieve the current Q-value
        current_q = self.q_function[self.state][action]
#         print 'Current Q = ' + str(current_q)
        
        ## Update the state variables after action
        ## (1) Traffic variables 
        new_inputs = self.env.sense(self)
        light = new_inputs['light']
        oncoming = new_inputs['oncoming']
        left = new_inputs['left']
        ## (2) Direction variables
        self.next_waypoint = self.planner.next_waypoint()
        deadline = self.env.get_deadline(self)
        if t == 1:
            self.gamma = 1 - float(4)/deadline
#             print self.gamma
#         location = self.env.agent_states[self]['location']
#         distance = self.env.compute_dist(location, destination)
#         heading = self.env.agent_states[self]['heading']
        
        ## Update the new state, which is a tuple of state variables
        self.state = (light, oncoming, left, self.next_waypoint)
        ## Get the best q_action in the new state
        new_action = self.select_action(state=self.state, is_current=False, t=t)    
        ## Get the new Q_value
        new_q = self.q_function[self.state][new_action]
        
        ## Update the Q-function
#         if (current_state == self.state) and (action is not None):
#             print (action, current_state, self.state)
#             sys.exit()
        current_alpha = 1 / math.sqrt(t+1)
#         current_alpha = self.alpha
        self.q_function[current_state][action] = (1 - current_alpha) * current_q + current_alpha * (reward + self.gamma * new_q)
#         print 'Updated Q = ' + str(self.q_function[current_state][action])
#         print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    ## TODO: Delete n_dummies, fw and progress in final submission
    ## Create a log file for the environment for each run
    log_filename = '../smartcab.log'
    fw = open(log_filename, 'w')
    n_trials = 100
    progress = ProgressBar(maxval=n_trials).start()
    env = Environment(n_dummies=3, fw=fw, progress=progress)  # create environment and add (3) dummy agents
    
    ## Create agent primary agent
    agent = env.create_agent(LearningAgent)  # create a learning agent
    env.set_primary_agent(agent, enforce_deadline=True)  # set agent to track
    
    # Now simulate it
    sim = Simulator(env, update_delay=0.01)  # reduce update_delay to speed up simulation
    start_time = time.time()
    sim.run(n_trials=n_trials)  # press Esc or close pygame window to quit
    progress.finish()
    runtime = round((time.time()-start_time) / 60, 2)
    runtime_str = 'Runtime = ' + str(runtime) + ' minutes\n'
    fw.write(runtime_str)
    fw.close() # close the log writer
    
    ## Show the runtime
    print '--- %s minutes ---' % runtime
    
    ## Compute the success trials
    success_trials = env.get_success_trials()
#     cumulative_rewards = env.get_cumulative_rewards()
#     for i in range(len(success_trials)):
#         print ((i+1), success_trials[i], cumulative_rewards[i])
    
    counter = 0
    success_count = 0
    for is_success in reversed(success_trials):
        if is_success:
            success_count += 1
        counter += 1
        if counter == 10:
            break
    if success_count >= 7:
        print 'Success!'
        ## Save the Q-table for later use
        datetime_int = int(calendar.timegm(time.gmtime()))
        out_filename = '../q_tables/q_table_' + str(datetime_int) + '.csv'
        f = open(out_filename, 'wb')
        writer = csv.writer(f)
        q_function = agent.get_q_function()
        for state, action_function in q_function.items():
            q_row = []
            q_row.append(state)
            for action in env.valid_actions:
                q_row.append(action_function[action])
            writer.writerow(q_row)
        f.close()
        print 'Written to file: ' + out_filename
    else:
        print 'Failure :('
    
if __name__ == '__main__':
    run()
