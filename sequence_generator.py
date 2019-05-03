import numpy as np
import random
from myutils import rename_state
from pyhsmm.util.general import rle

def gaussian_generator(mean, deviation, sample_size):
    data = np.array([np.random.normal(mean,deviation) for x in range(sample_size)])
    return data

STAGE_DICT = {chr(x): x - 97 for x in range(97,123)}

class Sequence:
    def __init__(self, n, alphabet = [], period = 0, type = None, p = None, params = None, mean = None , variance = None,
                 is_sorted = True, hsmm_model = None):
        self.n = n
        self.stateseq = None
        self.period = period

        if is_sorted == True:
            self.alphabet = sorted(alphabet)
        else:
            self.alphabet = alphabet

        self.mean = mean
        self.variance = variance
        # print(self.alphabet)
        self.params = params
        self.type = type
        # self.seq = np.zeros((n,),dtype=np.int64)
        self.sequence = []
        if type == 'period':
            self.periodic()
        if type == 'random':
            self.random(p)
        if type == 'signal':
            self.sequence, stages = self.signal_()
        if type == 'test_discret':
            self.test_discrete(params)
        if type == 'continue':
            self.continue_signal(mean,variance, params)
        if type == 'model':
            self.initialization_from_model(hsmm_model)

    def continue_signal(self, mean, variance, params):
        self.test_discrete(params)
        self.sequence = [ np.random.normal(mean[i],variance[i]) for i in self.stateseq]

    def random(self,p):
        h = np.zeros((len(p) + 1, 2))
        for i in range(len(p)+1):
            if i == 0:
                h[i,0] = 0; h[i,1] = p[i]
            else:
                if i == len(p):
                    h[i,0] = p[i-1]; h[i,1] = 1
                else:
                    h[i,0] = p[i-1]; h[i,1] = p[i]
        # print(h)

        for _ in range(self.n):
            r = random.uniform(0, 1)
            for i in range(h.shape[0]):
                if r >= h[i,0] and r < h[i,1]:
                    self.sequence += self.alphabet[i]
                    continue
    def generate_signal(self):
        self.continue_signal(self.mean, self.variance, self.params)
        return self.sequence

    def test_discrete(self,params=None):
        self.sequence = []
        if params == None:
            params = {'a': {'len': [2, 5], 'depend_on': False},
                      'b': {'len': [2, 10], 'depend_on': False},
                      'c': {'len': [2, 7], 'depend_on': False},
                      'd': {'len': [1, 5], 'depend_on': False },
                      'e': {'len': [1, 3], 'depend_on': True}}
        length = 0

        # print(params)
        for key, item in params.items():
            length += max(item['len'])
            assert item['len'][0] <= item['len'][1], 'Некорректный интервал [{};{}]'.format(item['len'][0], item['len'][1])

            
        count_cycle = round(self.n/length)

        is_first = True

        rest = self.n
        while rest>0:
            for s in self.alphabet:
                if is_first == True:   #Обработка для первого элемента списка
                    r = np.random.choice(range(params[s]['len'][0], params[s]['len'][1] + 1))
                    rest-=r
                    for _ in range(r):
                        self.sequence.append(s)
                    is_first = False
                else:
                    if params[s]['depend_on'] == self.sequence[-1]:
                        continue
                    elif params[s]['depend_on'] == 'randomly':
                        if np.random.uniform()<0.5: # какая вероятность
                            r = np.random.choice( range(params[s]['len'][0], params[s]['len'][1] + 1))
                            if rest < r:
                                for _ in range(rest):
                                    self.sequence.append(s)
                            else:
                                for _ in range(r):
                                    self.sequence.append(s)
                            rest-=r
                        else:
                            continue
                    else:
                        r = np.random.choice( range(params[s]['len'][0], params[s]['len'][1] + 1))
                        if rest < r:
                            for _ in range(rest):
                                self.sequence.append(s)
                        else:
                            for _ in range(r):
                                self.sequence.append(s)
                        rest-=r
        
        # for i in enumerate(range(count_cycle)):
        #     for s in self.alphabet:
        #         if is_first == True:   #Обработка для первого элемента списка
        #             r = np.random.choice(range(params[s]['len'][0], params[s]['len'][1] + 1))
        #             for _ in range(r):
        #                 self.sequence.append(s)
        #             is_first = False
        #         else:
        #             if params[s]['depend_on'] == self.sequence[-1]:
        #                 continue
        #             else:
        #                 r = np.random.choice( range(params[s]['len'][0], params[s]['len'][1] + 1))
        #                 for _ in range(r):
        #                     self.sequence.append(s)
        
        self.stateseq = [ STAGE_DICT[s] for s in self.sequence]
        self.n = len(self.sequence)

    def periodic(self):
        m = self.n // len(self.alphabet)
        rest = self.n % len(self.alphabet)
        for i in range(m):
            for k,s in enumerate(self.alphabet):
                self.sequence += self.alphabet[k]
            if i == m - 1:
                for k in range(rest):
                    self.sequence += self.alphabet[k]

    def get_slice(self,_sequence=None):
        """
        Получить случайно выбранную подпоследовательность из одного и того же символа с
        Returns:
        start : индекс начала подпоследовательности
        stop : индекс окончания
        с : символ       
        """
        if isinstance(_sequence, np.ndarray):
            m = np.random.choice(range(len(self.sequence)))
        else:
            if _sequence == None:
                m = np.random.choice(range(len(self.sequence)))
            else:
                m = np.random.choice(range(100,len(_sequence ) - 100))
        # m = len(s) - 1
        c = self.stateseq[m]
        start, stop = 0, 0
        i = 1 
        flag_1 = True
        flag_2 = True
        
        while((flag_1==True) or (flag_2 == True)):
            if flag_1 == True:
                if m-i-1 <= 0:
                    start = m-i+1
                    flag_1 = False
                else:
                    if self.stateseq[m - i]!=c:
                        start = m-i+1
                        flag_1 = False
            if flag_2 == True:
                if m + i - 1 >= len(self.stateseq) - 1:
                    stop = m + i
                    flag_2 = False
                else:
                    if self.stateseq[m+i]!=c:
                        stop = m + i
                        flag_2 = False
            i+=1
        # if m == len(self.sequence) - 1:
        #     print(start,stop)
        # print('f',m,start,stop)
        return start, stop, c   

    def get_abnormal(self, dtype = 'extension', extension_coef = 3, count_insert = 1, varience_coef = 2 , state = 0,
                     state_update = 1, count_segment = 1):
        """
        dtype : {extension, insert, varience, chain_violation}
        """
        if dtype == 'extension':
            return self._abnormal_extend_condition(extension_coef)
        elif dtype == 'insert':
            return self._get_abnormal_signal(count_insert)
        elif dtype == 'varience':
            return self._abnormal_increase_varience(state,varience_coef)
        elif dtype == 'chain_violation':
            return self._abnormal_chain_violation(state, state_update, count_segment)

    # Аномальная вставка 
    def _get_abnormal_signal(self, count = 1):
        x = self.sequence.copy()
        for _ in range(count):
            start, stop, c = self.get_slice()
            c-=1
            # print(c)
            # print(self.mean[c])
            # print(self.mean)
            x[start:stop] = np.random.normal(self.mean[c], self.variance[c], stop - start)
            # print(start,stop)
            # print(len(x))
        return x

    # Нарушение цепи маркова
    def _abnormal_chain_violation(self, state, state_update, count_segment):
        """
            count_segemnt:int - количество сегментов одного состояния для замены
        """
        x = np.array(self.sequence)
        states, pos = rle(self.stateseq)
        positions = np.cumsum(pos)
        index = np.where(np.array(states) == state)[0][-count_segment:]
        for it, ind in enumerate(index):
            if it == 0:
                indexs = np.arange(positions[ind - 1], positions[ind])
            else:
                indexs = np.concatenate((indexs,np.arange(positions[ind - 1], positions[ind])))

        x[indexs] = np.random.normal(self.mean[state_update], self.variance[state_update],
                                   len(indexs))
        return x
        
    def _abnormal_increase_varience(self, states, coef):
        assert len(states) < len(self.mean)
        x = np.array(self.sequence)
        for state in states:
            indx = np.where(np.array(self.stateseq) == state)[0]
            x[indx] = np.random.normal(self.mean[state], self.variance[state]*coef, len(indx))
        return x

    # Увеличение продолжительности состояния
    def _abnormal_extend_condition(self, coef ):
        """
            Возвращает сигнал, у которого увеличина продолжительность одного состояния
            в coef раз.
        """
        start, stop, c = self.get_slice(self.sequence)
        x = self.sequence.copy()
        size = int(np.mean(self.params[ self.alphabet[c] ]['len']) * coef)
        x[start:start+size] = np.random.normal(self.mean[c], self.variance[c], size)
        x[start+size:] = self.sequence[stop:]
        x = np.array(x[:self.n])
        if x.shape[0]>self.n:
            print('Проверить длину массива')
        else:
            return x

    # def anormal(self,p):
    #     x = self.sequence.copy()
    #     n = round(self.n*p)
    #     for i in range(n):
    #         index = round(np.random.uniform(0,self.n-1))
    #         x[index] = self.get_random_simbol()
    #     return x

    def to_int(self):
        seq = self.sequence.copy()
        x = []
        for s in seq:
            x+=[ord(s)%96]
        return x


    def get_random_simbol(self):
        s = round(np.random.uniform(len(self.alphabet))-1)
        return self.alphabet[s]

    def signal_(self):
        len_random_seq = 1
        data = np.array([])
        current_stage = 0
        seq_stages = []
        for _ in range(self.n):
            if current_stage == 0:
                a = np.random.uniform()
                if a >= 0 and a < 0.1:
                    temp = self.alphabet[0]
                    current_stage = 0
                elif a >= 0.1 and a < 0.5:
                    temp = self.alphabet[1]
                    current_stage = 1
                elif a >= 0.5 and a <= 1:
                    temp = self.alphabet[2]
                    current_stage = 2
            elif current_stage == 1:
                a = np.random.uniform()
                if a >= 0 and a <= 0.1:
                    temp = self.alphabet[0]
                    current_stage = 0
                elif a > 0.1 and a <= 0.2:
                    temp = self.alphabet[1]
                    current_stage = 1
                elif a > 0.2 and a <= 1:
                    temp = self.alphabet[2]
                    current_stage = 2
            else:
                a = np.random.uniform()
                if a >= 0 and a <= 0.1:
                    temp = self.alphabet[0]
                    current_stage = 0
                elif a > 0.1 and a <= 0.2:
                    temp = self.alphabet[1]
                    current_stage = 1
                elif a > 0.2 and a <= 1:
                    current_stage = 2
            if len(data) == 0:
                data = temp
            else:
                data = np.append(data, temp)
            seq_stages += [current_stage]
        return list(data), seq_stages

    def initialization_from_model(self, hsmm_model=None):
        """
            Инициализировать параметры распределения сигнала из модели.
            T : int - длина сигнала
        """
        assert len(hsmm_model.obs_distns[0].sigma[0]) <= 1
        mean = []
        sigma = []
        signal, stateseq = hsmm_model.generate(self.n)
        for i in range(len(hsmm_model.obs_distns)):
            mean+=[hsmm_model.obs_distns[i].mu[0]]
            sigma+=[np.sqrt(hsmm_model.obs_distns[i].sigma[0])]
        self.mean = mean
        self.variance = sigma
        self.sequence = signal
        self.stateseq = stateseq


class SemiMarkovSignal(Sequence):
    def __init__(self, _init_dist = None, _trans_matrix = None, _means = None, _variance = None , _N = None, _T = None, _dur_param = None):
        self.init_dist = np.array(_init_dist)
        self.trans_matrix = _trans_matrix
        self.mean = _means
        self.dur_param = _dur_param
        self.variance = _variance
        self.count_states = _N
        self.T = _T
        self.sequence = np.zeros((self.T))
        self.stateseq = np.zeros((self.T))
        #Start
        int_cumsum = np.concatenate((np.array([0]), self.init_dist)).cumsum()
        cur_len = 0
        while cur_len < self.T:
            if cur_len == 0:
                cur_state = self._get_state(int_cumsum)
                dur = np.random.poisson(self.dur_param[cur_state])
                self.sequence[cur_len:cur_len+dur] = np.random.normal(self.mean[cur_state], self.variance[cur_state], size = dur)
                self.stateseq[cur_len:cur_len+dur] = cur_state
                cur_len +=dur
            else:
                trans_cumsum = np.concatenate((np.array([0]),self.trans_matrix[cur_state])).cumsum()
                # print(trans_cumsum)
                cur_state = self._get_state(trans_cumsum)
                # print(cur_state)
                dur = np.random.poisson(self.dur_param[cur_state])
                if dur > self.T - cur_len:
                    self.sequence[cur_len:cur_len+dur] = np.random.normal(self.mean[cur_state], self.variance[cur_state], size =self.T - cur_len)
                else:
                    self.sequence[cur_len:cur_len+dur] = np.random.normal(self.mean[cur_state], self.variance[cur_state], size = dur)
                self.stateseq[cur_len:cur_len+dur] = cur_state
                cur_len +=dur

    def generate_signal(self):
        #Start
        int_cumsum = np.concatenate((np.array([0]), self.init_dist)).cumsum()
        cur_len = 0
        while cur_len < self.T:
            if cur_len == 0:
                cur_state = self._get_state(int_cumsum)
                dur = np.random.poisson(self.dur_param[cur_state])
                self.sequence[cur_len:cur_len+dur] = np.random.normal(self.mean[cur_state], self.variance[cur_state], size = dur)
                self.stateseq[cur_len:cur_len+dur] = cur_state
                cur_len +=dur
            else:
                trans_cumsum = np.concatenate((np.array([0]),self.trans_matrix[cur_state])).cumsum()
                # print(trans_cumsum)
                cur_state = self._get_state(trans_cumsum)
                # print(cur_state)
                dur = np.random.poisson(self.dur_param[cur_state])
                if dur > self.T - cur_len:
                    self.sequence[cur_len:cur_len+dur] = np.random.normal(self.mean[cur_state], self.variance[cur_state], size =self.T - cur_len)
                else:
                    self.sequence[cur_len:cur_len+dur] = np.random.normal(self.mean[cur_state], self.variance[cur_state], size = dur)
                self.stateseq[cur_len:cur_len+dur] = cur_state
                cur_len +=dur
        return self.sequence


    def _get_state(self,pi):
        rnd = np.random.random()
        # print(rnd)
        for i in range(len(pi)-1):
            if (rnd > pi[i]) and (rnd <= pi[i+1]):
                return i
        return i

    def get_abnormal(self, dtype='extension', extension_coef=3, count_insert=1, varience_coef=2, state=0,
                     state_update=1, count_segment=1):
        """
        dtype : {extension, insert, varience, chain_violation}
        """
        if dtype == 'extension':
            return self._abnormal_extend_condition(extension_coef)
        elif dtype == 'insert':
            return self._get_abnormal_signal(count_insert)
        elif dtype == 'varience':
            return self._abnormal_increase_varience(state, varience_coef)
        elif dtype == 'chain_violation':
            return self._abnormal_chain_violation(state, state_update, count_segment)

    # Аномальная вставка
    def _get_abnormal_signal(self, count=1):
        x = self.sequence.copy()
        for _ in range(count):
            start, stop, c = self.get_slice()
            c -= 1
            # print(c)
            # print(self.mean[c])
            # print(self.mean)
            x[start:stop] = np.random.normal(self.mean[c], self.variance[c], stop - start)
            # print(start,stop)
            # print(len(x))
        return x

    # Нарушение цепи маркова
    def _abnormal_chain_violation(self, state, state_update, count_segment):
        """
            count_segemnt:int - количество сегментов одного состояния для замены
        """
        x = np.array(self.sequence)
        states, pos = rle(self.stateseq)
        positions = np.cumsum(pos)
        index = np.where(np.array(states) == state)[0][-count_segment:]
        for it, ind in enumerate(index):
            if it == 0:
                indexs = np.arange(positions[ind - 1], positions[ind])
            else:
                indexs = np.concatenate((indexs, np.arange(positions[ind - 1], positions[ind])))

        x[indexs] = np.random.normal(self.mean[state_update], self.variance[state_update],
                                     len(indexs))
        return x

    def _abnormal_increase_varience(self, states, coef):
        assert len(states) < len(self.mean)
        x = np.array(self.sequence)
        for state in states:
            indx = np.where(np.array(self.stateseq) == state)[0]
            x[indx] = np.random.normal(self.mean[state], self.variance[state] * coef, len(indx))
        return x

    # Увеличение продолжительности состояния
    def _abnormal_extend_condition(self, coef):
        """
            Возвращает сигнал, у которого увеличина продолжительность одного состояния
            в coef раз.
        """
        start, stop, c = self.get_slice(self.sequence)
        c = int(c)
        x = self.sequence.copy()
        size = int(self.dur_param[c] * coef)
        if start+size >= self.T:
            x[start:start + size] = np.random.normal(self.mean[c], self.variance[c], self.T - start)

        else:
            # print(start, stop, size)
            x[start:start + size] = np.random.normal(self.mean[c], self.variance[c], size)
            x[(start + size):] = self.sequence[:len(x) - (start + size)]
            # x[(start + size):] = self.sequence[stop:stop+size]

        x = np.array(x[:self.T])
        if x.shape[0] > self.T:
            print('Проверить длину массива')
        else:
            return x


class Signal(Sequence):
    def __init__(self, n, count_stage, mean, varience, t ):
        super().__init__(n)
        self.n = n
        self.count_stage = count_stage
        self.mean = mean
        self.varience = varience
        self.t = t
        self.sequence, self.stateseq = self.create_signal()
    def create_signal (self):
        l = 0
        sequence = []
        stateseq = []
        current_stage = 0
        while(True):
            n = int(np.random.uniform(self.t[0], self.t[1]))
            l += n
            if l < self.n:
                stateseq = stateseq + [current_stage]*n
                sequence = sequence + np.random.normal(self.mean[current_stage], self.varience[current_stage], n).tolist()
                if current_stage < self.count_stage - 1:
                    current_stage+=1
                else:
                    current_stage=0
            else:
                n = self.n - len(stateseq)
                stateseq = stateseq + [current_stage] * n
                sequence = sequence + np.random.normal(self.mean[current_stage], self.varience[current_stage],
                                                       n).tolist()
                break
        return sequence, stateseq

class Continue_Signal():
    def __init__(self, _n, _mean, _varience):
        self.n = _n
        self.mean = _mean  
        self.varience = _varience


# def periodic(len, simbols):
#     step  = 5
#     seq = np.zeros([len,0])
#     k = 0
#     while (k>len):
#         for s in simbols:
#             seq[k] = [s; k++ for i in range(step)]
#     return seq
#
# periodic(10,['a','b'])

