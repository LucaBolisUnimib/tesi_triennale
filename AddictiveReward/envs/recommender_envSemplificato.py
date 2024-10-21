import numpy as np
import random

class Recommender_envSemplificato:
    LEARNING_RATE = 0.1
    REWARD_ACTION_CHOSEN = 1
    REWARD_ACTION_NOT_CHOSEN = -1
    ARM_A = 0
    ARM_B = 1

    # e-greedy e parameter
    epsilon = 0.10
    arms = 2
    reward = 0
    current_arm = 0

    def __init__(self):
        # Initialize bandit parameters
        self.q_bandit = np.zeros(self.arms) #Q_a
        self.AGENT_REWARD_AT_ARM = [10, (-2, 2)]

        # For statistical purposes
        self.rewards_action = np.zeros(self.arms)
        self.count_action = np.zeros(self.arms)
        self.arm_statistics = []    # arm_statistics[time step] = chosen arm at time_step

    def calculate_action(self, TRANSITION_PROBABILITIES):
        if random.uniform(0, 1) < self.epsilon: # take random action 
            self.current_arm = random.randint(0, self.arms - 1)
        else: # 1 - e probability 
            if(len(set(self.q_bandit))) == 1: # to solve tie situations 
                self.current_arm = random.randint(0, self.arms - 1)
            else:
                self.current_arm = np.argmax(self.q_bandit)

        self.count_action[self.current_arm] += 1
        self.set_probabilities(TRANSITION_PROBABILITIES)

    def update_estimate_rewards(self):
        Q_a = self.q_bandit[self.current_arm]
        self.q_bandit[self.current_arm] = Q_a + (self.LEARNING_RATE *(self.reward - Q_a))

    def set_probabilities(self, TRANSITION_PROBABILITIES):
        END_NEUTRAL_STATE = 3#7
        ADDICTED = 4#8
        S0 = 2#4
        aD = 4#8
        aW = 3

        # Set agent reward
        reward = 0
        if type(self.AGENT_REWARD_AT_ARM[self.current_arm]) is not tuple:
            reward = self.AGENT_REWARD_AT_ARM[self.current_arm]
        else:
            min, max = self.AGENT_REWARD_AT_ARM[self.current_arm]
            reward = round(random.uniform(min, max), 2)

        if(self.current_arm == self.ARM_A):
            #                                             H      N      ?   A
            TRANSITION_PROBABILITIES[END_NEUTRAL_STATE][aW] =  [0, 0,     1.0,   0,  0]
            TRANSITION_PROBABILITIES[END_NEUTRAL_STATE][aD] =  [0, 0,     0,     0,  1.0]
        else: #elif(self.current_arm == self.ARM_B):
            #                                             H      N        ?    A
            TRANSITION_PROBABILITIES[END_NEUTRAL_STATE][aW] =  [0, 0,     1.0,     0,   0]
            TRANSITION_PROBABILITIES[END_NEUTRAL_STATE][aD] =  [0, 0,     1.0,     0,   0]
    
    def action_chosen(self):
        self.rewards_action[self.current_arm] += self.REWARD_ACTION_CHOSEN
        self.reward = self.REWARD_ACTION_CHOSEN

        # Set agent reward
        reward = 0
        if type(self.AGENT_REWARD_AT_ARM[self.current_arm]) is not tuple:
            reward = self.AGENT_REWARD_AT_ARM[self.current_arm]
        else:
            min, max = self.AGENT_REWARD_AT_ARM[self.current_arm]
            reward = round(random.uniform(min, max), 2)
        return reward
    
    def action_not_chosen(self):
        self.rewards_action[self.current_arm] += self.REWARD_ACTION_NOT_CHOSEN
        self.reward = self.REWARD_ACTION_NOT_CHOSEN

    def reset_bandit(self):
        # Reset bandit parameters

        self.current_arm = 0
        self.q_bandit = np.zeros(self.arms)

        self.rewards_action = np.zeros(self.arms)
        self.count_action = np.zeros(self.arms)
        self.arm_statistics = []

    def add_arm_statistic(self):
        self.arm_statistics.append(self.current_arm)

    def get_qValues(self):
        return self.q_bandit
    
    def get_rewards(self):
        return self.rewards_action
    
    def get_arm(self):
        return self.current_arm

    def get_arm_statistics(self):
        return self.arm_statistics
    
    #e-greedy for stationary problem, it isn't used
    '''
    def update_Q_N(self):
        Q_a = self.q_bandit[self.current_arm]
        N_a = self.number_action[self.current_arm]
        self.q_bandit[self.current_arm] = Q_a + (1/N_a) *(self.R - Q_a) #R-Q(A)
    '''
    
    #UCB, but it isn't actually used
    '''
    def _calculate_action_old(self):
        # Calculate action based on bandit algorithm
        self._calculate_rew()
        temp = np.zeros(self.arms)
        for ar in range(self.arms):
            temp[ar] = (self.q_bandit[ar] + (self.c * math.sqrt(math.log(self.t) / self.number_action[ar] )))
       
        new_arm = np.argmax(temp)
            
        if self.current_arm != new_arm:
            self.t += 1
        self.current_arm = new_arm
        
        if random.uniform(0, 1) < self.epsilon: # epsilon greedy 
            self.current_arm = random.randint(0, 1)
    
    def _calculate_rew(self): #OK
        # Calculate reward for each arm
        for ar in range(self.arms):
            self.q_bandit[ar] = self.reward_action[ar] / self.number_action[ar] 
    '''
