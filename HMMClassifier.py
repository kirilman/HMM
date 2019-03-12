from pomegranate import *
import numpy as np
import json
import sequence_generator

class HHMClassifier:
    
    def __init__(self):
        self.models = []

    def add_model(self,model):
        self.models.append(model)

    def get_log_likelihood(self,X):
        """
            Определить формат X
        """
        score = [model.log_probability(X) for model in self.models]
        y_pred = np.argmax(score)
        return score, y_pred

class SignalManager:
    def __init__(self):
        # self.path_to_config = _path_to_config
        self.parameters = None
        self.generators = []

    def read_paramets(self,_path_to_config):
        with open(_path_to_config,'r') as file:
            pars_string = json.load(file)
            file.close()
        self.parameters = pars_string['array']

        for param in self.parameters:
            generator = sequence_generator.Sequence(param['N'], param['alpha'], type = param['type'],
                                 params = param['params'], mean = param['mean'], variance = param['variance'],
                                 is_sorted = param['sorted'])
            self.generators.append(generator)
    def get_signal_and_path(self,index):
        return self.generators[index].sequence, self.generators[index].path
            