import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mutation import *
from utils import *
from setting import *
import pickle

'''
Exhuastive search for small mask space (can be used to evaluate the evolutionary search result)
'''
# 48723 victim_1
victim = api.ModelSpec(
        matrix=[[0, 1, 0, 0, 0,0],
                [0, 0, 1, 0, 1,1],
                [0, 0, 0, 1, 0,1],
                [0, 0, 0, 0, 0,1],
                [0, 0, 0, 0, 0,1],
                [0, 0, 0, 0, 0,0]],  # output layer
        # Operations at the vertices of the module, matches order of matrix.
        ops=[INPUT, MAXPOOL3X3, MAXPOOL3X3, MAXPOOL3X3, MAXPOOL3X3,OUTPUT])
vict_flop, vict_test = info(victim)
print('res',vict_flop, vict_test)


mask_flops = []
mask_tests = []
population = []
i = 0
while True:
# for i in range(1):
    ori_spec = victim
    num_edge = ori_spec.original_matrix.sum()
    num_muta = 9 - num_edge
    ori_l = len(population)

    while num_muta > 0 or CONV1X1 in ori_spec.original_ops:
        # print(ori_spec.original_ops, ori_spec.original_matrix)
        spec = mutate_spec(victim,ori_spec)
        flag = check(spec, population)
        # print(flag)
        if flag == 1:
            # print('generate',i)
            population.append(spec)
            # print(spec.original_matrix,spec.original_ops)
            mask_flop, mask_test = info(spec)
            mask_flops.append(mask_flop)
            mask_tests.append(mask_test)
        num_edge = spec.original_matrix.sum()
        num_muta = 9 - num_edge
        ori_spec = spec
    l = len(population)
    if l > ori_l:
        i = 0
    else:
        i += 1
    if i > 300:
        print('Done')
        break

print(len(population))

## ================== save results =================
with open('./data/population1.pkl', 'wb') as f:
    pickle.dump(population, f)
np.save('./data/mask_tests1.npy', np.array(mask_tests))
np.save('./data/mask_flops1.npy', np.array(mask_flops))

## ============ find the best mask ==================
idx = np.argmin(mask_tests)
best_mask = population[idx]

print('Best mask spec \n', best_mask.original_matrix, best_mask.original_ops)
print('Info of best mask \n',mask_flops[idx],mask_tests[idx])
