
import sys
sys.path.append('/home/kirill/Projects/nir')
sys.path.append('/home/kirilman/Projects/nir/nir/')
import myutils
import sequence_generator as generator
import numpy as np
import matplotlib.pylab as plt
from pomegranate import *
from myutils import frequency_occurrence

plt.rcParams.update({'font.size':18 })
plt.rcParams.update({'figure.figsize':[10,5]})

N = 1000
alpha = ['a','b','c','d']
gen = generator.Sequence(N, alpha, type = 'test_discret',
                          params={'a': {'len': [25, 25], 'depend_on': False},
                                  'b': {'len': [25, 25], 'depend_on': False},
                                  'c': {'len': [15, 15], 'depend_on': False},
                                  'd': {'len': [25, 25], 'depend_on': False},
                                  'e': {'len': [15, 15], 'depend_on': False}
                                 },
                mean = [0, 0.5, 2, 4, 5] , variance = [0.03, 0.05, 0.1, 0.1, 0.14])

arr_anomal = []

for i in range(200):
    mask = np.random.choice(alpha +[False]*4,5)
#     print(mask)
    arr_anomal += [generator.Sequence(N, alpha, type = 'test_discret',
                      params={'a': {'len': [25, 25], 'depend_on': mask[0]},
                              'b': {'len': [25, 25], 'depend_on': mask[1]},
                              'd': {'len': [40, 40], 'depend_on': mask[2]},
                              'c': {'len': [15, 15], 'depend_on': mask[3]},
                              'e': {'len': [15, 15], 'depend_on': mask[4]}
                             },
                                
            mean = [0, 0.5, 2, 4, 5] , variance = [0.03, 0.05, 0.1, 0.1, 0.14])]
    print(i, sep=' ')
    print(mask)
train_signal = gen.sequence
fig = plt.figure(figsize = (20, 4))

labels = list(map(myutils.rename_state,gen.path))
plt.plot(train_signal)

labels = list(map(myutils.rename_state,gen.path))
model = HiddenMarkovModel.from_samples(DiscreteDistribution, n_components = 5, X = [train_signal])
log_norm = model.log_probability(train_signal)
fig = plt.figure(dpi = 100, figsize=(20,5))
count = 0
arr_log = []
numbers = []
for k,signal in enumerate(arr_anomal):
    lp = model.log_probability(signal.sequence)
    arr_log +=[lp]
    if lp > log_norm:
        count+=1
        numbers +=[k]

for k, signal in enumerate(arr_anomal):
    fig, ax = plt.subplots(2,1);
#     fig.figsize(25,4)
    fig.set_dpi(150)
    if k in numbers:
        ax[0].set_title('Detected as normal')
    ax[0].plot(train_signal,'g',label='Origin')
    ax[1].plot(signal.sequence,'r',label='Abnormal')
    # ax.plot(norm_signal,'b',label='Normal')  #Ошибка в цветах
    # ax.plot(an_signal,'r',label='Abnormal')
    ax[0].set_xlabel('Time',)
    ax[1].set_xlabel('Time',)
    ax[0].legend(loc=1)
    plt.tight_layout()
    plt.legend(loc=1)
#     plt.savefig('Plot/'+str(k)+'.svg',format="svg", dpi=240);
    plt.savefig('Plot/l_'+str(k)+'.png', dpi=240);
#     break
    plt.close('all')
    