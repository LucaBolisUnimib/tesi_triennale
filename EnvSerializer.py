import pickle
import numpy as np

class EnvSerializer:

    # [statoAgente, azioneAgente, nrewardAgente, 
    # [QvaluesAgente(matricer sxa)], raccomandazioneRecomender(A o B), [Qa_Recomender], rewardRecomender]
    value_at_step = [] # array that saves an array of valuable infrmation for each step of the simulation
    agents = [] #array in which every index is associated with an array of value_at_step 
    FILENAME = "dati.pkl"

    def __init__(self):
        self.value_at_step = []
        self.agents = []
        open(self.FILENAME, 'w').close()
    
    def add_value_at_step(self, val):
        self.value_at_step.append(val)
    
    def clean_value_at_step(self):
        self.value_at_step = []

    def serialize_data(self):
         with open(self.FILENAME,'ab') as f:
            pickle.dump(self.value_at_step, f)
