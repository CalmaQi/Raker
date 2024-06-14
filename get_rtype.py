import os
import argparse
import logging
import re
from sys import path
import ipdb
from numpy.lib.npyio import load

from warnings import simplefilter

from collections import defaultdict, Counter
import networkx as nx
from torch import rand
from tqdm import tqdm, trange
import json
import random
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import math
def analyze_relation_types(triplets):
    # Initialize dictionaries to store the tail entities for each head entity and vice versa
    relation_heads_tails = defaultdict(lambda: defaultdict(set))
    relation_tails_heads = defaultdict(lambda: defaultdict(set))
    head_relations = defaultdict(lambda: defaultdict(int))
    tail_relations = defaultdict(lambda: defaultdict(int))
    # Iterate over the triples
    for h, r, t in triplets:
        # Increment the count for the current relation type for the head entity
        head_relations[h][r] += 1
        # Increment the count for the current relation type for the tail entity
        head_relations[t][r] += 1
        relation_heads_tails[r][h].add(t)
        relation_tails_heads[r][t].add(h)

    # Initialize dictionaries to store the relation types
    relation_types = defaultdict(list)

    # Iterate over the relations
    for r in relation_heads_tails.keys():
        # Calculate the average number of tail entities for each head entity
        avg_tail_entities = sum(len(tails) for tails in relation_heads_tails[r].values()) / len(relation_heads_tails[r])
        # Calculate the average number of head entities for each tail entity
        avg_head_entities = sum(len(heads) for heads in relation_tails_heads[r].values()) / len(relation_tails_heads[r])

        # Determine the type of the relation
        if avg_tail_entities < 1.5 and avg_head_entities < 1.5:
            relation_type = '1-1'
        elif avg_tail_entities >= 1.5 and avg_head_entities < 1.5:
            relation_type = '1-N'
        elif avg_tail_entities < 1.5 and avg_head_entities >= 1.5:
            relation_type = 'N-1'
        else:
            relation_type = 'N-N'

        # Add the relation to the appropriate list
        relation_types[relation_type].append(r)

    return relation_types, head_relations   
def find_relation_type(relation_types, r):
    for relation_type, relations in relation_types.items():
        if r in relations:
            return relation_type
    # ipdb.set_trace()
    return '1-1'  

relation2id={}
relation_idx=0

relation2text = {}
with open(f'data/text/FB237/relation2text.txt') as fin:
    for l in fin:
        relation, text = l.strip().split('\t')
        relation2text[relation] = text
for i in relation2text:
    relation2id[i]=relation_idx
    relation_idx+=1
triples=[]
with open(f'data/fb237/train.txt') as fin:
    for l in fin:
        e1, r, e2 = l.strip().split('\t')
        triples.append([e1, r, e2]) 
rstrs={'1-1':0,'1-N':1,'N-1':2,'N-N':3}
  
r_type,_=analyze_relation_types(triples)
with open('r_type.txt','w')as f: 
    for i in relation2id:
        rstr=find_relation_type(r_type,i)
        f.write(str(relation2id[i])+'\t'+str(rstrs[rstr])+'\n')