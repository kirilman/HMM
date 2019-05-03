# from pomegranate import *
import numpy as np
import json
import sequence_generator
import pyhsmm
from pyhsmm.util.general import rle
import pyhsmm.basic.distributions as distributions
from pyhsmm.util.text import progprint_xrange

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


from multiprocessing import Pool

class hsmm_classifier():
    def __init__(self, N=5):
        self.models = []
        self.number_model = N

    def fit(self, data):
        pool = Pool(4)
        params = [(data, i + 10) for i in range(self.number_model)]
        self.models = pool.starmap(self.create_model, params)
        pool.close()
        pool.join()

    def log_likelihood(self, data):
        #         pool = Pool(self.number_model)
        #         return pool(self.models.log_likelihood, [(data,)*self.number_model])
        return np.array([m.log_likelihood(data) for m in self.models])

    def test(self):
        for m in self.models:
            print(m.generate(10, 1))

    def create_model(self, data, seed):
        np.random.seed(seed)
        obs_dim = 1
        dur_distns = []
        Nmax = 7
        #     L = 5
        #     obs_hypparams = {'alpha_0':np.zeros(L)+0.1,
        #                     'K':L,
        #                      'alphav_0':np.zeros(L)+0.1,
        #                      'alpha_mf':np.zeros(L)+0.1,
        #                     }

        obs_hypparams = {'mu_0': np.zeros(obs_dim),
                         'sigma_0': np.eye(obs_dim),
                         'kappa_0': 2,
                         'nu_0': obs_dim + 5}
        obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]

        dur_hypparams = {'alpha_0': 45,
                         'beta_0': 1}
        #     dur_distns +=[distributions.PoissonDuration(**dur_hypparams)]
        #     dur_hypparams = {'alpha_0':20,
        #                      'beta_0':1}
        #     dur_distns +=[distributions.PoissonDuration(**dur_hypparams)]
        #     dur_hypparams = {'alpha_0':30,
        #                      'beta_0':1}
        #     dur_distns +=[distributions.PoissonDuration(**dur_hypparams)]
        #     dur_hypparams = {'alpha_0':55,
        #                      'beta_0':1}
        #         dur_distns +=[distributions.PoissonDuration(**dur_hypparams)]

        #         dur_distns = [distributions.GeometricDuration(**dur_hypparams) for state in range(Nmax)]

        dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            alpha=6., gamma=2.,  # better to sample over these; see concentration-resampling.py
            init_state_concentration=6.,  # pretty inconsequential
            obs_distns=obs_distns,
            dur_distns=dur_distns)

        #     posteriormodel = pyhsmm.models.HSMM(
        #             alpha=6., # На что влияет
        # #             gamma=2., # better to sample over these; see concentration-resampling.py
        #             init_state_concentration=6., # pretty inconsequential
        #             obs_distns=obs_distns,
        #             dur_distns=dur_distns)

        posteriormodel.add_data(data)  # duration truncation speeds things up when it's possible
        for idx in progprint_xrange(150):
            posteriormodel.resample_model(1)
        return posteriormodel

    def get_models(self):
        return self.models

    def write_parametrs(filename):
        file = open(filename, 'w')
        for i, model in enumerate(classifiear.models):
            file.write('Модель = ' + str(i) + '\n')
            file.write('Cписок состояний: ' + str(model.used_states) + '\n')
            used_states = model.used_states
            used_states.sort()
            for k, dist in enumerate(model.obs_distns):
                if k in used_states:
                    file.write('Состояние: {} | mu = {}, sigma = {}, | lamda = {}'.format(k, dist.params['mu'],
                                                                                          np.sqrt(
                                                                                              dist.params['sigma'][0]),
                                                                                          model.dur_distns[k].params[
                                                                                              'lmbda']) + '\n')
                else:
                    continue
                    print('Состояние: {} , {}'.format(k, dist.params))
            file.write('Матрица переходов' + '\n')
            print(model.states_list[0].trans_matrix[[used_states]][:, used_states], '\n', '')
            file.write(str(model.states_list[0].trans_matrix[[used_states]][:, used_states]) + '\n')

            for i in used_states:
                for j in used_states:
                    file.write('{:.2F} '.format(model.states_list[0].trans_matrix[i, j]))
                file.write('\n')
            file.write('\n')
        file.close()