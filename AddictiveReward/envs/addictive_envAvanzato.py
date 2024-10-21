import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces
import random
from numpy.random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional
from .recommender_envAvanzato import Recommender_envAvanzato

class AddictiveEnv_Avanzato(gym.Env):
    def __init__(self):
        # PARAMS
        self.numero_stati = 23                      # Numero di stati + 1 
        self.numero_azioni = 9                      # Numero of actions
        self.S0 = 4                                 # Starting state
        self.reward_penalty_addictive_arm_a = -4    # Punishment end of Addictive Area
        self.C_penality = -1.2                      # Punishment in Addictive Area
        self.reward_addicted = 10                   # Reward at entering Addictive reward state
        self.reward_healty = 1                      # Reward 
        
        self.DINIT = 50 # Duration safe phase
        self.DDRUG = 9950 # In realtà nel paper è 1000

        self.HEALTHY = 1
        self.START_NEUTRAL_STATE = 2
        self.END_NEUTRAL_STATE = 7
        self.DRUG_STATE = 8
        self.START_AFTEREFFECTS_STATE = 9
        self.END_AFETEREFFECTS_STATE = 22

        ## ACTIONS
        #as2-7 -> 0-5
        self.START_ACTION_aS = 0
        self.END_ACTION_aS = 5
        self.aG = 6
        self.aW = 7
        self.aD = 8
        self.AR = ["as2", "as3", "as4", "as5", "as6", "as7", "aG", "aW", "aD"] # Non utilizzato (da me)
        
        # Observation space (20 stati 1 healty, 2-7 neutral 8-22 dipendenza)
        self.observation_space = spaces.Discrete(self.numero_stati) # non uso stato 0 
        # Actions spaces (9 azioni -> as 2-7, ag, aw, ad)
        self.action_space = spaces.Discrete(self.numero_azioni) 
        # stato iniziale: 4        
        self.state = self.S0
        # env_phase: 0-50 safe -> addictive deactivated
        # env_phase: 50-5000 safe -> addictive activated
        self.env_phase = 0

        # Initalize transition probabilities
        self.setTransitionProbabilities()

        # Initialize recommender
        # self.recommender = Recommender4arms()
        self.recommender = Recommender_envAvanzato()

    def setTransitionProbabilities(self):
        # Default probability and reward is [0, 0] for each triple
        self.TRANSITION_PROBABILITIES = [[[[0, 0] for k in range(self.numero_stati)] for j in range(self.numero_azioni)] for i in range(self.numero_stati)]
        
        # P(s=i|s=i,a=aSi) = 1.0, i neutral state
        # P(s=i+j|s=i,a=aS(i+j)) = 0.99, j=+1/-1, i neutral state, i+j neutral state
        # P(s=i|s=i, a=aS(i+j)) = 0.01, j=+1/-1, i neutral state, i+j neutral state
        # P(s=i+k|s=i,a=aS(i+k)) = 0.0001, k!=+1/-1, i neutral state, i+k neutral state
        # P(s=i|s=i,a=S(i+k)) = 0.9999, k!=+1/-1, i neutral state, i+k neutral state
        for state in range(self.START_NEUTRAL_STATE, self.END_NEUTRAL_STATE + 1):
            # aS2 is at index 0 of actions array
            aS_plus1 = state - self.START_NEUTRAL_STATE + 1
            as_minus1 = state - self.START_NEUTRAL_STATE - 1
            for action in range(self.START_ACTION_aS, self.END_ACTION_aS + 1):
                next_state = self.START_NEUTRAL_STATE + action
                if state == next_state:
                    self.TRANSITION_PROBABILITIES[state][action][next_state] = [1.0, 0]
                elif aS_plus1 in range(self.START_ACTION_aS, self.END_ACTION_aS + 1) and action == aS_plus1:
                    self.TRANSITION_PROBABILITIES[state][action][state] = [0.01, 0]
                    self.TRANSITION_PROBABILITIES[state][action][next_state] = [0.99, 0]
                elif as_minus1 in range(self.START_ACTION_aS, self.END_ACTION_aS + 1) and action == as_minus1:
                    self.TRANSITION_PROBABILITIES[state][action][state] = [0.01, 0]
                    self.TRANSITION_PROBABILITIES[state][action][next_state] = [0.99, 0]
                else:
                    self.TRANSITION_PROBABILITIES[state][action][next_state] = [0.0001, -0.3]
                    self.TRANSITION_PROBABILITIES[state][action][state] = [0.9999, 0]

        # P(s=i|s=i,a=aW) = 1.0, i neutral state
        for state in range(self.START_NEUTRAL_STATE, self.END_NEUTRAL_STATE + 1):
            self.TRANSITION_PROBABILITIES[state][self.aW][state] = [1.0, 0]

        # P(s=1|s=2,a=aG) = 1.0
        self.TRANSITION_PROBABILITIES[self.START_NEUTRAL_STATE][self.aG][self.HEALTHY] = [1.0, 0]

        # P(s=i|s=i,a=aG) = 1.0, i!=2 neutral state
        for state in range(self.START_NEUTRAL_STATE + 1, self.END_NEUTRAL_STATE + 1):
            self.TRANSITION_PROBABILITIES[state][self.aG][state] = [1.0, 0]

        # P(s=i|s=i,a=aD) = 1.0, i!=7 neutral state
        for state in range(self.START_NEUTRAL_STATE, self.END_NEUTRAL_STATE):
            self.TRANSITION_PROBABILITIES[state][self.aD][state] = [1.0, 0]

        # P(s=i|s=i,a=aG) = 0.999, i drug/aft state
        # P(s=4|s=i,a=aG) = 0.001, i drug/aft state
        # P(s=i|s=i,a=a(S*)) = 0.999, i drug/aft state
        # P(s=4|s=i,a=a(S*)) = 0.001, i drug/aft state
        for state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
            self.TRANSITION_PROBABILITIES[state][self.aG][state] = [0.999, self.C_penality]
            self.TRANSITION_PROBABILITIES[state][self.aG][self.S0] = [0.001, self.reward_penalty_addictive_arm_a]
            for neutral_action in range(self.START_ACTION_aS, self.END_ACTION_aS + 1):
                self.TRANSITION_PROBABILITIES[state][neutral_action][state] = [0.999, self.C_penality]
                self.TRANSITION_PROBABILITIES[state][neutral_action][self.S0] = [0.001, self.reward_penalty_addictive_arm_a]

        # P(s=j|s=i,a=aW) = 0.4995, i!=15 drug/aft state, j next or previous drug/aft state
        # P(s=4|s=i,a=aW) = 0.001, i!=15 drug/aft state
        for state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
            next_state = state + 1
            previous_state = state - 1
            if state != 15:
                if next_state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
                    self.TRANSITION_PROBABILITIES[state][self.aW][next_state] = [0.4995, self.C_penality]
                if previous_state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
                    self.TRANSITION_PROBABILITIES[state][self.aW][previous_state] = [0.4995, self.C_penality]
                self.TRANSITION_PROBABILITIES[state][self.aW][self.S0] = [0.001, self.reward_penalty_addictive_arm_a]
        
        self.TRANSITION_PROBABILITIES[self.DRUG_STATE][self.aW][self.END_AFETEREFFECTS_STATE] = [0.4995, self.C_penality]
        self.TRANSITION_PROBABILITIES[self.END_AFETEREFFECTS_STATE][self.aW][self.DRUG_STATE] = [0.4995, self.C_penality]

        # P(s=14/16|s=15,a=aW) = 0.2
        self.TRANSITION_PROBABILITIES[15][self.aW][14] = [0.2, self.C_penality]
        self.TRANSITION_PROBABILITIES[15][self.aW][16] = [0.2, self.C_penality]

        # P(s=4|s=15,a=aW) = 0.6
        self.TRANSITION_PROBABILITIES[15][self.aW][self.S0] = [0.6, self.reward_penalty_addictive_arm_a]

        # P(s=j|s=i,a=aD) = 0.745, i drug/aft state, j next drug/aft state
        # P(s=j|s=i,a=aD) = 0.245, i drug/aft state, j previous drug/aft state
        # P(s=4|s=i,a=aD) = 0.001, i drug/aft state
        for state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
            next_state = state + 1
            previous_state = state - 1
            if next_state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
                self.TRANSITION_PROBABILITIES[state][self.aD][next_state] = [0.745, self.C_penality]
            if previous_state in range(self.DRUG_STATE, self.END_AFETEREFFECTS_STATE + 1):
                self.TRANSITION_PROBABILITIES[state][self.aD][previous_state] = [0.245, self.C_penality]
            self.TRANSITION_PROBABILITIES[state][self.aD][self.S0] = [0.001, self.reward_penalty_addictive_arm_a]
        
        self.TRANSITION_PROBABILITIES[self.DRUG_STATE][self.aD][self.END_AFETEREFFECTS_STATE] = [0.245, self.C_penality]
        self.TRANSITION_PROBABILITIES[self.END_AFETEREFFECTS_STATE][self.aD][self.DRUG_STATE] = [0.745, self.C_penality]

        # P(s=4|s=1,a=aG) = 1.0
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aG][self.S0] = [1.0, self.reward_healty]

        # P(s=1|s=1,a=a(S*)) = 1.0
        for neutral_action in range(self.START_ACTION_aS, self.END_ACTION_aS + 1):
            self.TRANSITION_PROBABILITIES[self.HEALTHY][neutral_action][self.HEALTHY] = [1.0, 0]

        # P(s=1|s=1,a=aW) = 1.0
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aW][self.HEALTHY] = [1.0, 0]

        # P(s=1|s=1,a=aD) = 1.0
        self.TRANSITION_PROBABILITIES[self.HEALTHY][self.aD][self.HEALTHY] = [1.0, 0]
        # print(self.TRANSITION_PROBABILITIES)

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
        
        if self.state == self.END_NEUTRAL_STATE:
            if action == self.aD:
                self.recommender.action_chosen()
            else:
                self.recommender.action_not_chosen()
            self.recommender.update_estimate_rewards()
            self.recommender.calculate_action(self.TRANSITION_PROBABILITIES)

        self.state = next_state

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

        self.recommender.calculate_action(self.TRANSITION_PROBABILITIES)

        return self.get_state(), {}
            
    def close(self):
        pass