import time
import random
from collections import OrderedDict
from simulator import Simulator

class TrafficLight(object):
    """A traffic light that switches periodically."""
    
    valid_states = [True, False]  # True = NS open, False = EW open
    
    def __init__(self, state=None, period=None):
        self.state = state if state is not None else random.choice(self.valid_states)
        ## The random period of being in one state {NS, EW}
        self.period = period if period is not None else random.choice([3, 4, 5])
        self.last_updated = 0
    
    def reset(self):
        self.last_updated = 0
    
    def update(self, t):
        '''
        Switches the state {NS, EW} of the traffic light.
        '''
        if t - self.last_updated >= self.period:
            self.state = not self.state  # assuming state is boolean
            self.last_updated = t

class Environment(object):
    """An environment within which all agents interact with."""
    
    valid_actions = [None, 'forward', 'left', 'right']
    valid_inputs = {'light': TrafficLight.valid_states,
                    'oncoming': valid_actions,
                    'left': valid_actions,
                    'right': valid_actions}
    valid_headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # East, North, West, South
    
    def __init__(self, n_dummies=3, fw=None, progress=None):
        self.done = False
        self.t = 0
        self.agent_states = OrderedDict()
        self.status_text = ""
        self.fw = fw # the log file writer
        self.cumulative_reward = 0
        self.trial = 0 # the trial number
        self.progress = progress # the progress bar
        self.success_trials = [] # list to record outcomes of trials
        self.cumulative_rewards = []
        
        # Road network
        self.grid_size = (8, 6)  # (cols, rows)
        self.bounds = (1, 1, self.grid_size[0], self.grid_size[1])
        self.block_size = 100
        self.intersections = OrderedDict()
        self.roads = []
        ## Put a traffic light at each intersection
        for x in xrange(self.bounds[0], self.bounds[2] + 1):
            for y in xrange(self.bounds[1], self.bounds[3] + 1):
                self.intersections[(x, y)] = TrafficLight()  # a traffic light at each intersection
        
        ## Set equal length (1) for all segments
        for a in self.intersections:
            for b in self.intersections:
                if a == b:
                    continue
                if (abs(a[0] - b[0]) + abs(a[1] - b[1])) == 1:  # L1 distance = 1
                    self.roads.append((a, b))
        
        ## Create dummy agents
        ## A dummy agent has no destination nor deadline
        self.num_dummies = n_dummies  # no. of dummy agents
        for i in xrange(self.num_dummies):
            self.create_agent(DummyAgent)
        
        # Primary agent
        self.primary_agent = None  # to be set explicitly
        self.enforce_deadline = False
    
    def set_cumulative_reward(self, cumulative_reward=0):
        self.cumulative_reward = cumulative_reward
    
    def set_trial_number(self, trial=0):
        self.trial = trial
        self.progress.update(trial-1)
    
    def get_success_trials(self):
        return self.success_trials
    
    def get_cumulative_rewards(self):
        return self.cumulative_rewards
    
    def create_agent(self, agent_class, *args, **kwargs):
        agent = agent_class(self, *args, **kwargs)
        ## All agents initially head South?
        self.agent_states[agent] = {'location': random.choice(self.intersections.keys()), 'heading': (0, 1)}
        return agent
    
    def set_primary_agent(self, agent, enforce_deadline=False):
        self.primary_agent = agent
        self.enforce_deadline = enforce_deadline
    
    def reset(self):
        self.done = False
        self.t = 0
        
        # Reset traffic lights
        for traffic_light in self.intersections.itervalues():
            traffic_light.reset()
        
        # Pick random start (origin) and destination
        start = random.choice(self.intersections.keys())
        destination = random.choice(self.intersections.keys())
        
        # Ensure starting location and destination are not too close
        while self.compute_dist(start, destination) < 4:
            start = random.choice(self.intersections.keys())
            destination = random.choice(self.intersections.keys())
        
        ## Set random start heading (N, S, E, W)
        start_heading = random.choice(self.valid_headings)
        deadline = self.compute_dist(start, destination) * 5
#         print "Environment.reset(): Trial set up with start = {}, destination = {}, deadline = {}".format(start, destination, deadline)
        
        # Initialize the dummy and primary agents
        for agent in self.agent_states.iterkeys():
            self.agent_states[agent] = {
                'location': start if agent is self.primary_agent else random.choice(self.intersections.keys()),
                'heading': start_heading if agent is self.primary_agent else random.choice(self.valid_headings),
                'destination': destination if agent is self.primary_agent else None,
                'deadline': deadline if agent is self.primary_agent else None}
            agent.reset(destination=(destination if agent is self.primary_agent else None))
    
    def step(self):
#         print "Environment.step(): t = {}".format(self.t)  # [debug]
        # Update the traffic lights
        for intersection, traffic_light in self.intersections.iteritems():
            traffic_light.update(self.t)
        
        # Update the agents' observations (states)
        for agent in self.agent_states.iterkeys():
            agent.update(self.t)

        self.t += 1
        if self.primary_agent is not None:
            if self.enforce_deadline and self.agent_states[self.primary_agent]['deadline'] <= 0:
                self.done = True
                output_str = str(self.trial) + ". Environment.reset(): Primary agent could not reach destination within deadline!\n"
                output_str += 'Cumulative reward = ' + str(self.cumulative_reward)
#                 print output_str
                self.fw.write(output_str + '\n')
                ## Record the failure trial
                self.success_trials.append(False)
                self.cumulative_rewards.append(self.cumulative_reward)
            self.agent_states[self.primary_agent]['deadline'] -= 1 # decrement the deadline of primary agent
    
    def sense(self, agent):
        '''
        This is weird -- normally the agent senses the env, not the env senses it self for the agent!
        '''
        ## Make sure the agent is one of those created
        assert agent in self.agent_states, "Unknown agent!"
        
        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        ## Figure out the traffic light status {'red', 'green'}
        ## heading = {NS, EW}
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'
        
        # Figure out the oncoming, left, and right traffic
        oncoming = None
        left = None
        right = None
        for other_agent, other_state in self.agent_states.iteritems():
            if agent == other_agent or location != other_state['location'] or (heading[0] == other_state['heading'][0] and heading[1] == other_state['heading'][1]):
                continue # pass if other agent is this agent, or other location is this location, or?
            other_heading = other_agent.get_next_waypoint()
            if (heading[0] * other_state['heading'][0] + heading[1] * other_state['heading'][1]) == -1:
                if oncoming != 'left':  # we don't want to override oncoming == 'left'
                    oncoming = other_heading
            elif (heading[1] == other_state['heading'][0] and -heading[0] == other_state['heading'][1]):
                if right != 'forward' and right != 'left':  # we don't want to override right == 'forward or 'left'
                    right = other_heading
            else:
                if left != 'forward':  # we don't want to override left == 'forward'
                    left = other_heading
        
        return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
    
    def get_deadline(self, agent):
        return self.agent_states[agent]['deadline'] if agent is self.primary_agent else None
    
    def act(self, agent, action):
        assert agent in self.agent_states, "Unknown agent!"
        assert action in self.valid_actions, "Invalid action!"
        
        state = self.agent_states[agent] # the agent's current state
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'
        
        # Move the agent if within bounds and obey traffic rules
        ## Note that the prescribed action translates into the new heading
        reward = 0  # reward/penalty
        move_okay = True
        if action == 'forward':
            if light != 'green':
                move_okay = False
        elif action == 'left':
            if light == 'green':
                ## Can turn left on green light
                heading = (heading[1], -heading[0]) # update the heading tuple
            else:
                move_okay = False
        elif action == 'right':
            ## Can always turn right no matter what traffic light is (right of way has been handled by the agent)
            heading = (-heading[1], heading[0]) # update the heading tuple
        
        if action is not None:
            ## Update the current location (intersection)
            location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
                        (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
            #if self.bounds[0] <= location[0] <= self.bounds[2] and self.bounds[1] <= location[1] <= self.bounds[3]:  # bounded
            
            ## Update the location and heading
            state['location'] = location
            state['heading'] = heading
            
            if move_okay:
#                 ## Update the current location (intersection)
#                 location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
#                             (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
#                 #if self.bounds[0] <= location[0] <= self.bounds[2] and self.bounds[1] <= location[1] <= self.bounds[3]:  # bounded
#                 
#                 ## Update the location and heading
#                 state['location'] = location
#                 state['heading'] = heading
                ## Reward if action matches next waypoint
                reward = 2 if action == agent.get_next_waypoint() else 0.5
            else:
                ## TODO: Why is the move not executed (state unchanged)? Does the env bans illegal move?
                ## Penalize if action violates traffic rules
                reward = -1
        else:
            ## Reward for doing nothing -- this gives a high tendency to do nothing!
            reward = 1
        
        ## Check if destination has been reached (in time)
        if agent is self.primary_agent:
            if state['location'] == state['destination']:
                if state['deadline'] >= 0:
                    reward += 10  # BIG bonus
                    output_str = str(self.trial) + ". Environment.act(): Primary agent has reached destination!\n"  # [debug]
                    ## Record the success trial
                    self.success_trials.append(True)
                else:
                    output_str = str(self.trial) + ". Environment.act(): Primary agent has reached destination exceeding deadline!\n"  # [debug]
                    ## Record the failure trial
                    self.success_trials.append(False)
                
                self.cumulative_reward += reward
                output_str += 'Cumulative reward = ' + str(self.cumulative_reward)
#                 print output_str
                self.fw.write(output_str + '\n')
                self.cumulative_rewards.append(self.cumulative_reward)
                self.done = True
            else:
                self.cumulative_reward += reward
            
            self.status_text = "state: {}\naction: {}\nreward: {}".format(agent.get_state(), action, reward)
            #print "Environment.act() [POST]: location: {}, heading: {}, action: {}, reward: {}".format(location, heading, action, reward)  # [debug]
            
        return reward
    
    def compute_dist(self, a, b):
        """L1 distance between two points."""
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
class Agent(object):
    """Base class for all agents."""
    
    def __init__(self, env):
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.color = 'cyan'

    def reset(self, destination=None):
        pass

    def update(self, t):
        pass

    def get_state(self):
        return self.state

    def get_next_waypoint(self):
        return self.next_waypoint
    
    def set_params(self):
        pass

class DummyAgent(Agent):
    color_choices = ['blue', 'cyan', 'magenta', 'orange']
    
    def __init__(self, env):
        super(DummyAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.next_waypoint = random.choice(Environment.valid_actions[1:])
        self.color = random.choice(self.color_choices)
    
    def update(self, t):
        inputs = self.env.sense(self)
        action_okay = True
        
        ## Check if right of way is obeyed
        if self.next_waypoint == 'right': # wants to turn right
            ## Cannot turn right on red light if traffic on the left moves forward (goes straight)
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward': # wants to move forward
            if inputs['light'] == 'red': # can't move forward if there is red light
                action_okay = False
        elif self.next_waypoint == 'left': # wants to turn left
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                ## Cannot turn left if oncoming traffic moves forward or moves right
                action_okay = False
        
        action = None # stand still if next waypoint is invalid
        if action_okay: # set action equal to the next waypoint
            action = self.next_waypoint
            ## Generate a random next waypoint
            self.next_waypoint = random.choice(Environment.valid_actions[1:])
        reward = self.env.act(self, action)
        #print "DummyAgent.update(): t = {}, inputs = {}, action = {}, reward = {}".format(t, inputs, action, reward)  # [debug]
        #print "DummyAgent.update(): next_waypoint = {}".format(self.next_waypoint)  # [debug]
