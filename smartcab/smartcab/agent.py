import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns how to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        '''
        At each time step t, the agent:
        - Is given the next waypoint location (relative to its current location and direction)
        - Senses the intersection state (traffic light and presence of other vehicles)
        - Gets the current deadline value (time remaining)
        '''
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        ## The state variables
        inputs = self.env.sense(self)
        light = inputs['light']
        oncoming = inputs['oncoming']
        right = inputs['right']
        left = inputs['left']
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        
        # TODO: Select action according to your policy
#         action = self.next_waypoint
        action = random.choice(self.env.valid_actions)
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for finite number of trials."""
    # Set up environment and agent
    env = Environment()  # create environment and add (3) dummy agents
    ## Create agent primary agent
    agent = env.create_agent(LearningAgent)  # create a learning agent
    env.set_primary_agent(agent, enforce_deadline=False)  # set agent to track
    
    # Now simulate it
    sim = Simulator(env, update_delay=0.80)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
