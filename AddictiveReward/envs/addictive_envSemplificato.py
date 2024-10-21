import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces
import random
from numpy.random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
import math

from .recommender_envSemplificato import Recommender_envSemplificato

class AddictiveEnv_Semplificato(gym.Env):

    NEUTRAL = 2
    HEALTHY = 1
    ADDICTED = 3  
    AFFTEREFFECTS = 4
    as2 = 0
    as3 = 1
    aG = 2
    aW = 3
    aD = 4

    def __init__(self):
        # Initialize environment parameters

        # Number of states
        self.numero_stati = 5  # 0:x 1:H 2:N 3:A 4:C
        
        # Number of actions
        self.numero_azioni = 5  # as2 as3 aG aW aD
        
        # Set Neutral state
        self.S0 = 2

        # Set rewards
        self.reward_healty = 1
        self.reward_addicted = 10
        self.reward_penalty_addictive_arm_a = -4
        self.reward_penalty_addictive_arm_b = -1
        self.C_penality = -2.5  # Reward for standing still in aftereffects
        
        # Define observation and action space
        self.observation_space = spaces.Discrete(self.numero_stati)  # Not using state 0 
        self.action_space = spaces.Discrete(self.numero_azioni)         
        self.state = self.S0

        # Environment phase
        # 0-50 safe -> addictive deactivated
        # 50-1000 safe -> addictive activated
        self.env_phase = 0 
        self.DINIT = 50
        self.DDRUG = 4950

        # Initalize transition probabilities
        self.setTransitionProbabilities()

        # Initialize recommender
        #self.recommender = Recommender4arms()
        self.recommender = Recommender_envSemplificato()
    
    def setTransitionProbabilities(self):
        # TRANSITION_PROBABILITIES[state][action][next_state] is the probability to go to the next_state from the state by executing action
        # TODO: In realtà si dovrebbero cambiare le probabilità in base alla fase in cui si è, ovvero la fase init, drug, ...
        
        self.TRANSITION_PROBABILITIES = np.zeros([self.numero_stati, self.numero_azioni, self.numero_stati])
        print(self.HEALTHY, self.NEUTRAL, self.AFFTEREFFECTS)
        #                                                           H     N     ?   A
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.as2] = [0, 1.0,  0,    0,  0]
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.as3] = [0, 1.0,  0,    0,  0]
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aG] =  [0, 0,    1.0,  0,  0]
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aW] =  [0, 1.0,  0,    0,  0]
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aD] =  [0, 1.0,  0,    0,  0]

        #                                                           H     N     ?     A
        self.TRANSITION_PROBABILITIES[self.NEUTRAL][self.as2] = [0, 0,    1.0,  0,    0]
        self.TRANSITION_PROBABILITIES[self.NEUTRAL][self.as3] = [0, 0,    0,  1.0,    0]
        self.TRANSITION_PROBABILITIES[self.NEUTRAL][self.aG] =  [0, 1.0,  0,    0,    0]
        self.TRANSITION_PROBABILITIES[self.NEUTRAL][self.aW] =  [0, 0,    1.0,  0,    0]
        self.TRANSITION_PROBABILITIES[self.NEUTRAL][self.aD] =  [0, 0,    1.0,  0,  0]

        # The transition probabilities for the ? state are set according to the action of the recommender for the drug phase
        #                                                            H      N   ?       A
        self.TRANSITION_PROBABILITIES[self.ADDICTED][self.as2] = [0, 0,     1.0,  0,      0]
        self.TRANSITION_PROBABILITIES[self.ADDICTED][self.as3] = [0, 0,     0,    1.0,    0]
        self.TRANSITION_PROBABILITIES[self.ADDICTED][self.aG] =  [0, 0,     0,    1.0,    0]
        # These two rows are used only during the init phase, when addiction is deactivated
        # INUTILE (forse), dato che questi vengono da subito cambiati dal recommender
        self.TRANSITION_PROBABILITIES[self.ADDICTED][self.aW] =  [0, 0,     0,  1.0,    0]
        self.TRANSITION_PROBABILITIES[self.ADDICTED][self.aD] =  [0, 0,     0,  1.0,    0]
        
        #                                                                 H   N       ?   A
        self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.as2] = [0, 0,  0,      0,  1.0]
        self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.as3] = [0, 0,  0,      0,  1.0]
        self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.aG] =  [0, 0,  0.15,   0,  0.85]
        self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.aW] =  [0, 0,  0,  0,  1.0]
        self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.aD] =  [0, 0,  0,      0,  1.0]
        # print(self.TRANSITION_PROBABILITIES)
     
    def get_rewards(self):
        # return self.reward_action
        return self.recommender.get_rewards()
        
    def get_statistics(self):
        return self.recommender.get_arm_statistics()
            
    def get_recommender(self):
        return self.recommender
    
    def get_iter(self):
        return self.DINIT + self.DDRUG
        
    def get_state(self):
        return self.state

    def get_next_state(self, action):
        state = self.get_state()
        probabilities = self.TRANSITION_PROBABILITIES[state][action]
        next_state = random.choices(np.arange(self.numero_azioni), weights=probabilities)[0]
        return next_state

    # Execute one time step within the environment
    def step(self, action):
        reward = 0

        next_state = self.get_next_state(action)

        if self.state == self.HEALTHY:
            if next_state == self.NEUTRAL:
                self.state = self.S0
                reward = self.reward_healty

        elif self.state == self.NEUTRAL:
            self.state = next_state
        
        elif self.state == self.ADDICTED:
            if action == self.aD:
                reward = self.recommender.action_chosen()
            else:
                self.recommender.action_not_chosen()
            self.recommender.update_estimate_rewards()
            self.recommender.calculate_action(self.TRANSITION_PROBABILITIES)
            self.state = next_state
            # Under the initial pre-drug phase (dinit = 50 steps), the agent does not receive any reward 
            # or negative outcome by entering the drug-related and aftereffects area, but a moderate 
            # reward is assigned (Rg = 1) by accessing the healthy reward state.
            if self.env_phase < self.DINIT:
                reward = 0
                
        elif self.state == self.AFFTEREFFECTS:
            if next_state == self.NEUTRAL:
                reward = self.reward_penalty_addictive_arm_a
            elif next_state == self.AFFTEREFFECTS:
                reward = self.C_penality
            self.state = next_state
            # Under the initial pre-drug phase (dinit = 50 steps), the agent does not receive any reward 
            # or negative outcome by entering the drug-related and aftereffects area, but a moderate 
            # reward is assigned (Rg = 1) by accessing the healthy reward state.
            if self.env_phase < self.DINIT:
                reward = 0
         
        if (self.env_phase == self.DDRUG + self.DINIT):
            terminated = True
        else:
            terminated = False
            self.env_phase += 1  

        self.recommender.add_arm_statistic()
        # print(reward)
        return self.get_state(), reward, terminated, False, {}
        
    def render(self):
        pass
            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset the environment
        super().reset(seed=seed)
        self.recommender.reset_bandit()
        self.state = self.S0
        self.env_phase = 0 

        self.recommender.calculate_action(self.TRANSITION_PROBABILITIES)

        return self.get_state(), {}
            
    def close(self):
        pass