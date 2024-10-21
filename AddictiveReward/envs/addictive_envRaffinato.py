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

from .recommender_envRaffinato import Recommender_envRaffinato

class AddictiveEnv_Raffinato(gym.Env):

    aG = 0
    aW = 1
    aD = 2
    RN = 0
    def __init__(self):
        # Initialize environment parameters

        # Number of states
        self.numero_stati = 12  # 0:x 1:H 2:N 3-6:Rs 7-10:Rl 11:A
        
        # Number of actions
        self.numero_azioni = 3  # aG aW aD
        
        # Set Neutral state
        self.S0 = 2

        # Set rewards
        self.reward_healty = 1
        self.reward_addicted = 10
        self.reward_penalty_addictive_arm_a = -4
        self.reward_penalty_addictive_arm_b = -1
        self.C_penality = -1.2  # Reward for standing still in aftereffects
        
        # Define observation and action space
        self.observation_space = spaces.Discrete(self.numero_stati)  # Not using state 0 
        self.action_space = spaces.Discrete(self.numero_azioni)         
        self.state = self.S0

        # Environment phase
        # 0-50 safe -> addictive deactivated
        # 50-1000 safe -> addictive activated
        self.env_phase = 0 
        self.DINIT = 50
        self.DDRUG = 9950
        
        # Initialize recommender
        self.recommender = Recommender_envRaffinato()
        self.last_state = 0

        self.REC_SHORT_LAYER = 1
        self.HEALTHY = 1
        self.NEUTRAL = 2
        self.START_REC_SHORT = 3
        self.END_REC_SHORT = self.START_REC_SHORT + (self.recommender.arms) * self.REC_SHORT_LAYER - 1
        # print(self.END_REC_SHORT)
        self.START_REC_LONG = self.END_REC_SHORT + 1
        self.END_REC_LONG = self.START_REC_LONG + self.recommender.arms - 1
        self.AFFTEREFFECTS = self.END_REC_LONG + 1

        # Initalize transition probabilities
        self.setTransitionProbabilities()

    def setTransitionProbabilities(self):
        # TRANSITION_PROBABILITIES[state][action][next_state] is the [probability, reward] to go to the next_state from the state by executing action

        self.TRANSITION_PROBABILITIES = [[[[0, 0] for k in range(self.numero_stati)] for j in range(self.numero_azioni)] for i in range(self.numero_stati)]

        # Neutral state
        for action in range(self.numero_azioni):
            if action == self.aG:
                self.TRANSITION_PROBABILITIES[self.S0][self.aG][self.HEALTHY] = [1.0, 0]
            elif action != self.aD:
                self.TRANSITION_PROBABILITIES[self.S0][action][self.S0] = [1.0, 0]

        # Fake Healthy state
        for action in range(self.numero_azioni):
            if action == self.aG:
                self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aG][self.RN] = [1.0, 0]
            elif action == self.aD:
                self.TRANSITION_PROBABILITIES[self.HEALTHY][action][self.S0] = [1.0, 0]
            else:
                self.TRANSITION_PROBABILITIES[self.HEALTHY][action][self.HEALTHY] = [1.0, 0]

        # Real Healthy state
        for action in range(self.numero_azioni):
            if action == self.aG:
                self.TRANSITION_PROBABILITIES[self.RN][self.aG][self.S0] = [1.0, 1]
            else:
                self.TRANSITION_PROBABILITIES[self.RN][action][self.RN] = [1.0, 0]
        
        # Aftereffects state
        # The probability to exit the aftereffects state is 0.05 when action aG is performed
        for action in range(self.numero_azioni):
            if action == self.aG:
                self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.aG][self.AFFTEREFFECTS] = [0.95, -1.2]
                self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][self.aG][self.S0] = [0.05, -3.5]
            else:
                self.TRANSITION_PROBABILITIES[self.AFFTEREFFECTS][action][self.AFFTEREFFECTS] = [1.0, -1.2]

        # Rec short exit
        for state in range(self.START_REC_SHORT, self.END_REC_SHORT + 1):
            self.TRANSITION_PROBABILITIES[state][self.aG][self.S0] = [1.0, -1.1]
            self.TRANSITION_PROBABILITIES[state][self.aW][self.S0] = [1.0, -1.1]

        # Rec long exit
        for state in range(self.START_REC_LONG, self.END_REC_LONG + 1):
            for action in range(self.numero_azioni):
                self.TRANSITION_PROBABILITIES[state][action][self.S0] = [0.2, -3.2]
                self.TRANSITION_PROBABILITIES[state][action][self.AFFTEREFFECTS] = [0.8, -3.2]

    def get_rewards(self):
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
        probabilities, rewards = list(zip(*self.TRANSITION_PROBABILITIES[state][action]))
        next_state = random.choices(np.arange(self.numero_stati), weights=probabilities)[0]
        return (rewards[next_state], next_state)
    
    def step(self, action):
        reward = 0

        reward, next_state = self.get_next_state(action)
        
        if self.state == self.NEUTRAL or (self.state >= self.START_REC_SHORT and self.state <= self.END_REC_SHORT):
            if action == self.aD:
                self.recommender.action_chosen()
            else:
                self.recommender.action_not_chosen()
            
        if self.get_state() == self.NEUTRAL:
            self.recommender.update(self.last_state)
            self.last_state = self.NEUTRAL

        self.last_state = max(self.last_state, self.get_state())
        
        self.state = next_state
        if self.get_state() == self.NEUTRAL:
            self.recommender.calculate_action(self.get_state(), self.TRANSITION_PROBABILITIES)  


        if (self.env_phase == self.DDRUG + self.DINIT):
            terminated = True
        else:
            terminated = False
            self.env_phase += 1  

        self.recommender.add_arm_statistic()
        return self.get_state(), reward, terminated, False, {}

    def render(self):
        pass
  
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset the environment
        super().reset(seed=seed)
        self.recommender.reset_bandit()
        self.state = self.S0
        self.env_phase = 0 
        self.last_state = self.S0

        self.recommender.calculate_action(self.get_state(), self.TRANSITION_PROBABILITIES)

        return self.get_state(), {}
            
    def close(self):
        pass