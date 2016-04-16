import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns how to drive in the smartcab world."""

    def __init__(self, env, init_value=0, gamma=0.95, alpha=0.10, epsilon=0.10):
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
        self.alpha = alpha
        ## Parameter of the epsilon-greedy action selection strategy
        self.epsilon = epsilon
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
    
    def update(self, t):
        '''
        At each time step t, the agent:
        - Is given the next waypoint (relative to its current location and direction)
        - Senses the intersection state (traffic light and presence of other vehicles)
        - Gets the current deadline value (time remaining)
        '''
        ## The destination trying to reach
        destination = self.env.agent_states[self]['destination']
        
        ## Observe the current state variables
        ## (1) Traffic variables
        inputs = self.env.sense(self)
        light = inputs['light']
        oncoming = inputs['oncoming']
#         right = inputs['right']
        left = inputs['left']
        ## (2) Direction variables
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self)
        location = self.env.agent_states[self]['location']
        distance = self.env.compute_dist(location, destination)
        heading = self.env.agent_states[self]['heading']
        
        ## Update the current observed state
        self.state = (light, oncoming, left,
                      deadline, self.next_waypoint, distance, heading)
        
        ## TODO: Implement the epsilon-greedy action selection that selects best-valued action in this state
        ## with probability (1 - epsilon) and a random action with probability epsilon.
        if self.state in self.q_function:
            action_function = self.q_function[self.state]
            ## TODO: Find the action that has the highest value
            
            rand_action = random.choice(self.env.valid_actions)
            action = rand_action
        else:
            action = random.choice(self.env.valid_actions)        
        
        ## Execute action, get reward and new state
        reward = self.env.act(self, action)
        
        ## Update the state variables after action
        ## (1) Traffic variables 
        new_inputs = self.env.sense(self)
        light = new_inputs['light']
        oncoming = new_inputs['oncoming']
        left = new_inputs['left']
        ## (2) Direction variables
        self.next_waypoint = self.planner.next_waypoint()
        deadline = self.env.get_deadline(self)
        location = self.env.agent_states[self]['location']
        distance = self.env.compute_dist(location, destination)
        heading = self.env.agent_states[self]['heading']
        
        ## Update the new state, which is a tuple of state variables
        self.state = (light, oncoming, left,
                      deadline, self.next_waypoint, distance, heading)
        
        # TODO: Learn policy based on state, action, reward
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    ## TODO: Delete n_dummies in final submission
    env = Environment(n_dummies=3)  # create environment and add (3) dummy agents
    ## Create agent primary agent
    agent = env.create_agent(LearningAgent)  # create a learning agent
    env.set_primary_agent(agent, enforce_deadline=False)  # set agent to track
    
    # Now simulate it
    sim = Simulator(env, update_delay=0.80)  # reduce update_delay to speed up simulation
    ## Each trial is a distinct game?
    sim.run(n_trials=2)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
