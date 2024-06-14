# Stratified Sampling, keep all relations
import numpy as np
import ipdb
from collections import defaultdict
import random
import math


dataset = 'testgraph'
example_ids = []
relations = []
rel2eids = defaultdict(list)
entities=[]
num_random=49
data = []
pos_triple=defaultdict(list)
test_triple=[]
ranking_head=[]
ranking_tail=[]
with open(f'./data/{dataset}/train.txt') as fin:
    for i, l in enumerate(fin):
        e1, r, e2 = l.strip().split('\t')
        pos_triple[(e1,r)].append(e2)
        pos_triple[(r,e2)].append(e1)
        entities.append(e1)
        entities.append(e2)
        

with open(f'./data/{dataset}/test.txt') as fin:
    for i, l in enumerate(fin):
        e1, r, e2 = l.strip().split('\t')
        ranking_tail.append([e1,r,e2])
        ranking_head.append([e1,r,e2])
        neg_tail = set(entities) - set([e1,e2])-set(pos_triple[(e1,r)])
        neg_head = set(entities) - set([e1,e2])-set(pos_triple[(r,e2)])      
        neg_tail = random.sample(neg_tail, num_random)
        neg_head= random.sample(neg_head, num_random)
        for ii in neg_head:
            ranking_head.append([ii,r,e2])
        for ii in neg_tail:
            ranking_tail.append([e1,r,ii])
        test_triple.append([e1,r,e2])
with open (f'./data/{dataset}/ranking_head.txt','w') as fin:
    for i in ranking_head:
        fin.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\n')
with open (f'./data/{dataset}/ranking_tail.txt','w') as fin:
    for i in ranking_tail:
        fin.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\n')    

               
# with open(f'./data/{dataset}/test.txt') as fin:
#     for i, l in enumerate(fin):
#         e1, r, e2 = l.strip().split('\t')
#         example_ids.append(i)
#         relations.append(r)
#         data.append(l)
#         rel2eids[r].append(i)

# num_samples = 5000
# sample_ratio = float(num_samples) / len(example_ids)

# sampled_eids = []

# for r, eids in rel2eids.items():
#     sample_num_this_r = max(int(round(sample_ratio * len(eids))), 1)
#     sampled_eids.extend(random.sample(eids, sample_num_this_r))

# sampled_eids = sorted(sampled_eids)


# with open(f'./data/{dataset}/train_{num_samples}.txt', 'w') as fout:
#     for eid in sampled_eids:
#         fout.write(data[eid])
        
