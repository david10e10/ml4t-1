"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand


class QLearner(object):
    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=200, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states

        # Instatiate the actions possible array
        self.actions_possible = np.array(range(self.num_actions)) # Create an array of the amount of actions possible

        # Instantiate NDArray T, r
        self.t = np.empty([0,4], dtype=int)
        self.t_c = np.full((num_states, num_actions, num_states, 1), .00001)
        self.r_array = np.random.uniform(low=-1.0, high=1.0, size=(num_states, num_actions)) #TODO Really random vals?

        # Instantiate variables
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.s = 0
        self.a = 0

        # Initialize Q as a num_states x num_states numpy matrix w/ random vals -1.0 to 1.0
        self.q = np.random.uniform(low=-1.0, high=1.0, size=(num_states,num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s

        # If the random number is > than our probability, use policy, otherwise, pick a random action
        if (np.random.randint(100) > self.rar*100):
            action = self.actions_possible[np.argmax(self.q[self.s, self.actions_possible])] # Select the action based on the q value that is max'd
        else:
            action = rand.randint(0, self.num_actions - 1) # Original Action Code, simply a random number
        self.rar = self.rar * self.radr

        self.a = action

        if self.verbose: print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # Update Q with new value of
        #self.q[self.s, action] = (1-self.alpha)*self.q[self.s, action] + self.alpha*(r + self.gamma * max(self.q[s_prime, self.actions_possible]))
        self.q[self.s, self.a] = (1-self.alpha)*self.q[self.s, self.a] + self.alpha*(r + self.gamma * max(self.q[s_prime, self.actions_possible]))

        # If the random number is > than our probability, use policy, otherwise, pick a random action
        if (np.random.randint(100) > self.rar*100):
            action = self.actions_possible[np.argmax(self.q[s_prime, self.actions_possible])] # Select the action based on the q value that is max'd
        else:
            action = rand.randint(0, self.num_actions - 1) # Original Action Code, simply a random number
        self.rar = self.rar * self.radr

        # Update T
        self.t = np.concatenate((self.t, np.array([[self.s, self.a, s_prime, r]])), axis=0) #self.a or action
        # self.t_c[self.s, self.a, s_prime] = self.t_c[self.s, self.a, s_prime] + 1
        #>>> b = np.concatenate((a, np.array([[1,2,3,4]])), axis=0) #TODO Delete

        # Update R
        self.r_array[self.s, self.a] = (1 - self.alpha) * self.r_array[self.s, self.a] + self.alpha * (r)

        # #TODO non-vectorized
        # # Iterate over i n_dyna times
        # for i in range(self.dyna):
        #     # s_dyna = np.random.randint(self.num_states)
        #     # a_dyna = np.random.randint(self.num_actions)
        #
        #     s_dyna = self.t[:,0][np.random.randint(self.t[:,0].size)]
        #     a_dyna = self.t[(self.t[:,0]==s_dyna),1][np.random.randint(self.t[(self.t[:,0]==s_dyna),1].size)]
        #
        #     s_prime_dyna = (self.t[(self.t[:,0]==s_dyna)&(self.t[:,1]==a_dyna)])[:,2][np.random.randint((self.t[(self.t[:,0]==s_dyna)&(self.t[:,1]==a_dyna)])[:,2].size)]
        #     #t_dyna = (self.t_c[s_dyna, a_dyna, s_prime_dyna] / self.t_c[s_dyna, a_dyna, :])[0]
        #     # r_dyna = self.t[(self.t[:,0]==s_dyna)&(self.t[:,1]==a_dyna)]
        #     r_dyna = self.r_array[s_dyna, a_dyna]
        #     self.q[s_dyna, a_dyna] = (1-self.alpha)*self.q[s_dyna, a_dyna] + self.alpha*(r_dyna + self.gamma * max(self.q[s_prime_dyna, self.actions_possible]))

        #TODO vectorized
        # Select S and A arrays from the s,a samples in T
        s_dyna_arr = np.random.choice(self.t[:,0], size=self.dyna)
        a_dyna_arr = np.random.choice(self.t[:,1], size=self.dyna)
        s_prime_dyna_arr = np.random.choice(self.t[np.in1d(self.t[:,0],s_dyna_arr)&np.in1d(self.t[:,1],a_dyna_arr),2], size = self.dyna)
        r_dyna_arr = self.r_array[s_dyna_arr,a_dyna_arr]
        tuple_dyna_arr = np.concatenate([np.transpose([s_dyna_arr]), np.transpose([a_dyna_arr]), np.transpose([s_prime_dyna_arr]), np.transpose([r_dyna_arr])], axis=1)
        # Iterate over i n_dyna times
        for i in range(tuple_dyna_arr.shape[0]):
            # self.q[s_dyna, a_dyna] = (1-self.alpha)*self.q[s_dyna, a_dyna] + self.alpha*(r_dyna + self.gamma * max(self.q[s_prime_dyna, self.actions_possible]))
            self.q[tuple_dyna_arr[i,0], tuple_dyna_arr[i,1]] = (1-self.alpha)*self.q[tuple_dyna_arr[i,0], tuple_dyna_arr[i,1]] + self.alpha*(tuple_dyna_arr[i,3] + self.gamma * max(self.q[tuple_dyna_arr[i,2], self.actions_possible]))

        # Select an r based on <random s, random a>
        # Select a semi-random s' based on probability of <random s, random a>
        # Update Q based on <random s, random a, semi random s', R[s,a]>
        # May need to fix the order of this jazz


        if self.verbose: print "s =", s_prime, "a =", action, "r =", r

        # Update s (= s') and a (= a) before returning action
        self.a = action
        self.s = s_prime

        return action


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
