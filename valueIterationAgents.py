# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        
        for iteration in range(self.iterations):  
            current_iteration_values = util.Counter()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                values = []

                for action in actions:
                    values.append(self.computeQValueFromValues(state, action))
                
                current_iteration_values[state] = max(values, default=0)
            
            self.values = current_iteration_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        if self.mdp.isTerminal(state):
            return None
        
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        current_action_sum = 0

        for transition in transitions:
            next_state, transition_prob = transition
            reward_transition = self.mdp.getReward(state, action, next_state)
            v_next_state = self.values[next_state]
            curr_v_term = transition_prob * (reward_transition + self.discount * v_next_state)
            current_action_sum += curr_v_term
        
        return current_action_sum

    def computeBellmanValueFromValues(self, state):
        actions = self.mdp.getPossibleActions(state)
        values = []

        for action in actions:
            values.append(self.computeQValueFromValues(state, action))
        
        return max(values, default=0)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        
        actions = self.mdp.getPossibleActions(state)
        values = []

        for action in actions:
            values.append(self.computeQValueFromValues(state, action))
        
        max_idx = max(range(len(values)), key=values.__getitem__)
        return actions[max_idx]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        counter = 0

        for iteration in range(self.iterations): 
            state = states[counter]            
            self.values[state] = self.computeBellmanValueFromValues(state)
            counter = (counter + 1) % len(states)
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        predecessors = {}

        for state in states:
            predecessors[state] = set()

        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for next_state, transition_prob in transitions:
                    if transition_prob > 0:
                        predecessors[next_state].add(state)

        priority_queue = util.PriorityQueue()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            
            actions = self.mdp.getPossibleActions(state)
            
            diff = abs(self.values[state] - self.computeBellmanValueFromValues(state))
            priority_queue.push(state, -diff)
        
        for iteration in range(self.iterations):
            if priority_queue.isEmpty():
                break
            
            state = priority_queue.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = self.computeBellmanValueFromValues(state)
            
            for predecessor in predecessors[state]:
                diff = abs(self.values[predecessor] - self.computeBellmanValueFromValues(predecessor))
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)


