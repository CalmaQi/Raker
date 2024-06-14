"""
Prepare data for BERTRL
"""
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
def flatten(list):
    return [_ for sublist in list for _ in sublist]


def show_path(path, entity2text):
    return [entity2text[e] for e in path]


def get_valid_paths(exclude_edge, paths):
    e1, e2, r_ind = exclude_edge
    # edge paths
    valid_paths = []
    for path in paths:
        if not any((e1, e2, r_ind) == edge for edge in path):
            valid_paths.append(path)
    return valid_paths


def is_in_G(e1, r, e2, G):
    if e1 in G and e2 in G and e2 in G[e1] and r in [r_dict['relation'] for r_dict in G[e1][e2].values()]:
        return True
    return False


def construct_local_entity2text(subgraph_entities, entity2text):
    return entity2text
    local_text2entities = defaultdict(list)
    for e in subgraph_entities:
        local_text2entities[entity2text[e]].append(e)
    if len(local_text2entities) == len(subgraph_entities):
        local_entity2text = entity2text
    else:
        local_entity2text = {}
        for e in subgraph_entities:
            e_text = entity2text[e]
            if len(local_text2entities[e_text]) == 1:
                local_entity2text[e] = e_text
            else:
                local_entity2text[e] = e_text + ' ' + \
                    str(local_text2entities[e_text].index(e))
    return local_entity2text


def construct_subgraph_text(G, subgraph_entities, entity2text, relation2text, excluded=None):
    G_text = []
    
    # deal with repeat entities
    for e1, e2, r_dict in G.subgraph(subgraph_entities).edges.data():
        if (e1, r_dict['relation'], e2) == excluded:
            continue
        e1_text, e2_text = entity2text[e1], entity2text[e2]
        rtmp=r_dict['relation']
        flag=0
        if r_dict['relation'].startswith('inv-'):
            rtmp=r_dict['relation'][4:]
            flag=1
        r_text = relation2text[rtmp]
        #调整头尾实体顺序
        if flag:
            edge_text = f'{e2_text} {r_text} {e1_text};'          
        else:  
            edge_text = f'{e1_text} {r_text} {e2_text};'
        
        if f'{e2_text} {r_text} {e1_text};' not in G_text and f'{e1_text} {r_text} {e2_text};' not in G_text:
            G_text.append(edge_text)

    return G_text

def path_form_text(biG,e1path,commonrel,entity2text,relation2text,num,tflag=0):
    G_text = [] 
    for i in e1path:
        rel=biG[i[0][0]][i[0][1]][i[0][2]]['relation']
        flag=0
        if rel.startswith('inv-'):
            rel=rel[4:]
            flag=1
        r_text = relation2text[rel]
        e1_text, e2_text = entity2text[i[0][0]], entity2text[i[0][1]]
        if  commonrel:
            if  rel in commonrel:
                # 对于带有inv逆关系的调整头尾实体顺序
                if flag:
                    edge_text = f'{e2_text} {r_text} {e1_text};'
                else:
                    edge_text = f'{e1_text} {r_text} {e2_text};'
                G_text.append(edge_text)           
        else :
            if flag:
                edge_text = f'{e2_text} {r_text} {e1_text};'
            else:
                edge_text = f'{e1_text} {r_text} {e2_text};'
            G_text.append(edge_text)
    if  commonrel and len(G_text)<num and tflag==1:
        
        for i in e1path:
            rel=biG[i[0][0]][i[0][1]][i[0][2]]['relation']
            flag=0
            if rel.startswith('inv-'):
                rel=rel[4:]
                flag=1
            r_text = relation2text[rel]
            e1_text, e2_text = entity2text[i[0][0]], entity2text[i[0][1]]
            if flag:
                edge_text = f'{e2_text} {r_text} {e1_text};'
            else:
                edge_text = f'{e1_text} {r_text} {e2_text};'
            if rel not in commonrel:
                G_text.append(edge_text)
            
    G_text=random.sample(G_text, min(len(G_text), num)) 
    return G_text
    # return ['']
def construct_commonrel_text(biG,e1path,e2path,commonrel,entity2text,relation2text,num1=6,num2=6):
                  
    G_text=path_form_text(biG,e1path,commonrel,entity2text,relation2text,num1)
    G_text2=path_form_text(biG,e2path,commonrel,entity2text,relation2text,num2)
    G_text.extend(G_text2)  
    # if not G_text:
    #     G_text=[' ']
    
    # if not G_text[0]:
    #     G_text=[' ']
    # return []
    return G_text

def construct_rel_text(biG,relation_ht_relation,head_relations,relation_heads_tails,
                       relation_tails_heads,entity2text,relation2text,e1,e2,r,one_to_many=None,many_to_one=None,topk=17,threadshold=0.1):
    # Sort the dictionary by value
    # return [' ']
    # if r == '/people/person/profession':
    #     ipdb.set_trace()
        
    flag_on=[' ']
    head_relation = sorted(relation_ht_relation[r][0].items(), key=lambda item: item[1], reverse=True)
    head_relation=[item[0] for item in head_relation[:topk] if item[1] >= threadshold]
    # if len(head_relation)>5:
    #     two_thirds_length = max(int(len(head_relation) * 2 / 3),5)
    #     head_relation = head_relation[:two_thirds_length]
    tail_relation = sorted(relation_ht_relation[r][1].items(), key=lambda item: item[1], reverse=True)
    tail_relation=[item[0] for item in tail_relation[:topk] if item[1] >= threadshold]
    # if len(tail_relation)>5:
    #     two_thirds_length = max(int(len(tail_relation) * 2 / 3),5)
    #     tail_relation = tail_relation[:two_thirds_length]
    
    hr=head_relations[e1].keys()
    tr=head_relations[e2].keys()
    
    if r in one_to_many:
        if e1 in one_to_many[r]:
            pass
            # for i in one_to_many[r][e1]:
            #     one_to_many[r][e1][i]=relation_ht_relation[r][1][i]
            # tail_relation=sorted(one_to_many[r][e1].items(), key=lambda item: item[1], reverse=True)
            # tail_relation=[item[0] for item in tail_relation[:topk]]
    
        else:#假设只有这些头实体
            # pass
            tail_relation=[]
            tail_relation=tail_relation
            
    if r in many_to_one:
        if e2 in many_to_one[r]:
            pass
            # for i in many_to_one[r][e2]:
            #     one_to_many[r][e2][i]=relation_ht_relation[r][0][i]
            # head_relation=sorted(many_to_one[r][e2].items(), key=lambda item: item[1], reverse=True)
            # head_relation=[item[0] for item in head_relation[:topk]]
        else:
            head_relation=[]
            head_relation=head_relation
            
    filter_hr=[]
    filter_tr=[]
    # for rel in head_relation:
    #     if rel in hr:
    #         filter_hr.append(rel)
    # for rel in tail_relation:
    #     if rel in tr:
    #         filter_tr.append(rel)  
    ##better
    for rel in hr:
        if rel in head_relation:
            filter_hr.append(rel)
    for rel in tr:
        if rel in tail_relation:
            filter_tr.append(rel)
    def get_triplet(h,filter_hr,relation_heads_tails,relation_tails_heads):
        triplets=[]
        for rel in filter_hr:
            if h in relation_heads_tails[rel]:
                for t_set in relation_heads_tails[rel][h]:
                    triplets.append((h,rel,t_set))
            
            elif h in relation_tails_heads[rel]:
                for h_set in relation_tails_heads[rel][h]:
                    triplets.append((h_set,rel,h))

            # ipdb.set_trace()
            
        return triplets                  
    if filter_hr and filter_tr:
        h_triplet=get_triplet(e1,filter_hr,relation_heads_tails,relation_tails_heads)
        t_triplet=get_triplet(e2,filter_tr,relation_heads_tails,relation_tails_heads)
    else:
        return flag_on
    # if r in one_to_many:
    #     h_triplet=h_triplet[:2]
    # if r in many_to_one:
    #     t_triplet=t_triplet[:2]
    # ipdb.set_trace()
    result=list(set(h_triplet+t_triplet))
    # result=random.sample(result,min(11, len(result)))
    # ipdb.set_trace()
    if (e1,r,e2) in result:
        result.remove((e1,r,e2))
    # print(len(result))
    def get_sequence(result,entity2text,relation2text):
        # Initialize an empty list to store the text sequences
        text_sequences = [' ']
        # if not result:
        #     ipdb.set_trace()
        # Iterate over the triples in the result
        for head, relation, tail in result:
            # Convert the head, relation, and tail to text using the entity2text and relation2text dictionaries
            head_text = entity2text[head]
            relation_text = relation2text[relation]
            tail_text = entity2text[tail]
            # Combine the head text, relation text, and tail text into a single text sequence
            text_sequence = f"{head_text} {relation_text} {tail_text};"
            
            # Add the text sequence to the list of text sequences
            text_sequences.append(text_sequence)
        # text_sequences=random.sample(text_sequences,min(len(text_sequences),topk))import random
        # random.seed(42)  # You can use any number as the seed
        # random.shuffle(text_sequences)
        text_sequences=' '.join(text_sequences) 
        return [text_sequences]
        # Now, text_sequences is a list of text sequences corresponding to the triples in the result
    text_sequences=get_sequence(result,entity2text,relation2text)
    # ipdb.set_trace()

    return text_sequences

def construct_paths_text(biG, valid_paths, entity2text, relation2text, edge_pattern='{} {} {};', conclusion_relation=None, params=None):
    downsample, use_reversed_order, sort_by_len = params.downsample, params.use_reversed_order, params.sort_by_len
    # edge_pattern = '“ {} ” {} “ {} ”;'
    # edge_pattern = '{} {} {};'
    # construct text from valid edge paths
    if sort_by_len:
        valid_paths = sorted(valid_paths, key=lambda x: len(x))

    G_text = []
    relation_paths = []
    rel2eids = defaultdict(list)
    for i, path in enumerate(valid_paths):
        relation_path = [conclusion_relation]
        for j, (e1, e2, r_ind) in enumerate(path):
            r_dict = biG[e1][e2][r_ind]
            e1_text, e2_text = entity2text[e1], entity2text[e2]
            r = r_dict['relation']
            if r.startswith('inv-'):
                if not use_reversed_order:
                    r_text = relation2text[r[4:]]
                    edge_text = edge_pattern.format(e2_text, r_text, e1_text)
                else:  # reversed order
                    r_text = 'inv- ' + relation2text[r[4:]]
                    edge_text = edge_pattern.format(e1_text, r_text, e2_text)
            else:
                r_text = relation2text[r]
                edge_text = edge_pattern.format(e1_text, r_text, e2_text)
            relation_path.append(r)

            G_text.append(edge_text)
        relation_path = tuple(relation_path)
        relation_paths.append(relation_path)
        rel2eids[relation_path].append(i)
        G_text.append('[SEP]')  # including a [SEP] at the end

    G_text = G_text[:-1]  # exluce last [SEP]
    if not G_text:
        return G_text

    G_text = ' '.join(G_text).split(' [SEP] ')

    if downsample:  # downsample the repeated relation paths
        sampled_eids = []
        for _, eids in rel2eids.items():
            sample_num_this_r = min(len(eids), 1)
            sampled_eids.extend(random.sample(eids, sample_num_this_r))
        G_text = [G_text[eid] for eid in sampled_eids]

    return G_text

def generate_subgraph(biG,e1_pos,e2_pos,r_pos,set_type,r_pos_ind,entity2text,relation2text):
    neighbornum=1
    e1_neigh_to_dis = nx.single_source_shortest_path_length(biG, e1_pos, cutoff=neighbornum) # 3 
    e2_neigh_to_dis = nx.single_source_shortest_path_length(biG, e2_pos, cutoff=neighbornum)
    e1_neigh=[i for i in e1_neigh_to_dis.keys() if i is not e1_pos]
    e2_neigh=[i for i in e2_neigh_to_dis.keys() if i is not e2_pos]
    e1_neigh=random.sample(e1_neigh, min(len(e1_neigh), 3)) # neg = 10
    e2_neigh=random.sample(e2_neigh, min(len(e2_neigh), 3)) # neg = 10
    
    allsubpath=[]
    # subgraph_entities_pos = flatten([edge[:2] for edge in flatten(valid_paths)])
    # subgraph_entities_pos = list(set(subgraph_entities_pos))
    # local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
    
    for tmp in e1_neigh:
        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_pos, target=tmp, cutoff=neighbornum)]
        if set_type == 'train':
            valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)
        else:
            valid_paths=paths
        if valid_paths:
            allsubpath.append(valid_paths[0])
    for tmp in e2_neigh:
        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e2_pos, target=tmp, cutoff=neighbornum)]
        if set_type == 'train':
            valid_paths = get_valid_paths((e2_pos, e1_pos, r_pos_ind), paths)
        else:
            valid_paths=paths
        if valid_paths:
            allsubpath.append(valid_paths[0])   
    subgraph_entities_pos = flatten([edge[:2] for edge in flatten(allsubpath)])
    subgraph_entities_pos = list(set(subgraph_entities_pos))
    local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
    G_text_pos_edges = construct_subgraph_text(
        biG, subgraph_entities_pos, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
    G_text_pos_edges = ['  '.join(G_text_pos_edges)]
    return G_text_pos_edges[0]
            # ipdb.set_trace()






def generate_bert_input_from_scratch(biG, set_type, triples, params=None):
    entity2text, relation2text = params.entity2text, params.relation2text
        #增加
    relation2id={}
    relation_idx=0
    for i in relation2text:
        relation2id[i]=relation_idx
        relation_idx+=1
   
    entitylist=[i for i in biG]
    # e2longtext=json.loads('data/text/FB15K')
    en2text={}
    count_valid=[0,0,0,0,0,0]
    pos_support=[]
    neg_support=[]
    import json
    # with open ('data/text/FB15K/entities.json') as f:
    if params.dataset.startswith('WN18RR'):
        with open ('data/text/wn18rr/entities.json') as f:
            e2longtext=json.loads(f.read())
    elif  params.dataset.startswith('fb') or params.dataset.startswith('testgraph'):
        with open ('data/text/FB15K/entities.json') as f:   
            e2longtext=json.loads(f.read())
    elif  params.dataset.startswith('nell'):
        e2longtext=entity2text
        with open ('./entity2textOpenAI.txt') as f:
            for line in f:
                entity,text=line.split("\t")
                text=text.rstrip('\n')
                en2text[entity]=' '.join(text.split()[:30])
    else:
        e2longtext=entity2text
        with open ('data/text/'+params.dataset+'/entity2text.txt') as f:
            for line in f:
                entity,text=line.split("\t")
                text=text.rstrip('\n')
                en2text[entity]=' '.join(text.split()[:30])
        # en2text=e2longtext
        # for i in en2text:
        #     en2text[i]=''
    for i in e2longtext:
        if params.dataset.startswith('WN18RR'):
            entity2text[i['entity_id']]=i['entity'].split('_')[0]+i['entity'].split('_')[-1]
            # entity2text[i['entity_id']]=i['entity'].split('_')[0]+', '+i['entity_desc']
            
            # en2text[i['entity_id']]=entity2text[i['entity_id']]+' : '+i['entity_desc']
            en2text[i['entity_id']]=i['entity_desc']
            
        elif  params.dataset.startswith('fb') or params.dataset.startswith('testgraph'):
            en2text[i['entity_id']]=i['entity_desc']
            # en2text[i['entity_id']]=''
            
        # entity2text[i['entity_id']]=' '.join(i['entity'].split('_')[:-2])
        
        # entity2text[i['entity_id']]=i['entity']
        
        # en2text[i['entity_id']]=''
        #增加
        
        
    # question_pattern = 'Question: {} {} what ? Is the correct answer {} ? {} {}'
    question_pattern = 'Question: {} {} what ? Is the correct answer {}  ? '
    # question_pattern = 'Question: Is {} {} {} ? '
    # question_pattern= '{} {} Question: {} {} what ? Is the correct answer {}  ? '
    
    
    # context_pattern=''
    valid_paths_cnter = Counter()
    fout = open(f'{params.bertrl_data_dir}/{set_type}.tsv', 'w') 
    fout.write('# Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
    foutcontext = open(f'{params.bertrl_data_dir}/{set_type}context.tsv', 'w')
    foutcontext.write('Triple\tContext\tNeighbornum\trid\n')
    visual_nei=open('visual.txt','w')
    
    num_pos_examples = len(triples[set_type]['pos'])
    num_neg_samples_per_link = len(triples[set_type]['neg']) // num_pos_examples
    seen_neg = set()
    
      # 使用TF-IDF算法来过滤出那些在实体周围普遍存在的关系。
    # TF-IDF是一种统计方法，用于评估一个词对于一个文件集或一个语料库中的一个文件的重要程度。
    # 在这个场景中，我们可以将每个实体看作一个"文件"，将每个关系看作一个"词"。
   
    def normalize_head_relations(head_relations):
        # Initialize a dictionary to store the normalized relations for each head entity
        normalized_head_relations = defaultdict(lambda: defaultdict(float))

        # Iterate over the head entities
        for h, relations in head_relations.items():
            # Calculate the total count of relations for the current head entity
            total_count = sum(relations.values())

            # Normalize the count of each relation
            for r, count in relations.items():
                normalized_head_relations[h][r] = count / total_count

        return normalized_head_relations
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
        return None  
    def calculate_relation_proportion(triplets, target_relation):
        relation_count = sum(1 for _, r, _ in triplets if r == target_relation)
        total_triplets = len(triplets)
        return relation_count / total_triplets
    
    def get_rel(biG,triplets):
        relation_types,head_relations=analyze_relation_types(triplets)
        n_head_relations=normalize_head_relations(head_relations)
        headlist=set()
        taillist=set()
        relation_heads_tails = defaultdict(lambda: defaultdict(set))
        relation_tails_heads = defaultdict(lambda: defaultdict(set))
        relation_ht_relation = defaultdict(list)
        one_to_one=defaultdict(lambda: defaultdict(list))
        one_to_many=defaultdict(lambda: defaultdict(list))
        many_to_one=defaultdict(lambda: defaultdict(list))
        many_to_many=defaultdict(lambda: defaultdict(list))
        # relation_heads_relation
        # Iterate over the triples
        for h, r, t in triplets:
            headlist.add(h)
            taillist.add(t)
            relation_heads_tails[r][h].add(t)
            relation_tails_heads[r][t].add(h)
        # Now, selected_relations contains the relations that meet the criteria
        # Calculate the total number of head entities
        total_heads = len(n_head_relations)
        total_tails=len(n_head_relations)
        num_relation=len(relation_heads_tails)
        
        for r in relation_heads_tails.keys():   
            
            hr = defaultdict(float)
            tr = defaultdict(float)
            r_type=find_relation_type(relation_types,r)
            for h in relation_heads_tails[r].keys():
                h_relations = n_head_relations[h]
                for rel, freq in h_relations.items():
                    # if r =='/people/person/profession' and rel =='/people/cause_of_death/people':
                    #     ipdb.set_trace()
                    # Calculate TF
                    tf = freq 
                    # Calculate IDF
                    idf = math.log(total_heads / len([h for h in n_head_relations if rel in n_head_relations[h]]))
                    # Calculate TF-IDF
                    hr[rel] += tf * idf
                
            # Calculate the total number of tail entities

            for t_set in relation_heads_tails[r].values():
                for t in t_set:
                    t_relations = n_head_relations[t]
                    for rel, freq in t_relations.items():
                        # Calculate TF
                        tf = freq
                        # Calculate IDF
                        # idf = math.log(sum([n_tail_relations[h][rel] for h in n_tail_relations if rel in n_tail_relations[h]]) / tf)
                        
                        idf = math.log(total_tails / len([t for t in n_head_relations if rel in n_head_relations[t]]))
                        # Calculate TF-IDF
                        tr[rel] += tf * idf

            if r_type=='1-N':
                for h in relation_heads_tails[r].keys(): 
                    one_to_many_hr = defaultdict(float)
                    for t in relation_heads_tails[r][h]:
                        t_relations = n_head_relations[t]
                        for rel, freq in t_relations.items():
                            # Calculate TF
                            tf = freq
                            idf = math.log(total_tails / len([t for t in n_head_relations if rel in n_head_relations[t]]))
                            # Calculate TF-IDF
                            one_to_many_hr[rel] += tf * idf
                    one_to_many[r][h]=one_to_many_hr
            elif r_type=='N-1':
                for t in relation_tails_heads[r].keys(): 
                    many_to_one_tr = defaultdict(float)
                    for h in relation_tails_heads[r][t]:  
                        h_relations = n_head_relations[h]
                        for rel, freq in h_relations.items():
                            # Calculate TF
                            tf = freq
                            idf = math.log(total_heads / len([h for h in n_head_relations if rel in n_head_relations[h]]))
                            # Calculate TF-IDF
                            many_to_one_tr[rel] += tf * idf
                    many_to_one[r][t]=many_to_one_tr
            # ipdb.set_trace()
            relation_ht_relation[r].append(hr)    
            relation_ht_relation[r].append(tr)  

        # 假设你的字典如下：
        
        def norm_fre(relation_ht_relation, ht):
            for rel in relation_ht_relation:      
                # 提取字典中的值
                values = np.array(list(relation_ht_relation[rel][ht].values()))
                # 计算最小值和最大值
                min_val = np.min(values)
                max_val = np.max(values)
                # 检查值的数量
                if len(values) > 3:
                    # 进行最小-最大归一化
                    normalized_values = (values - min_val) / (max_val - min_val) if max_val != min_val else values
                else:
                    normalized_values = values
                # 将归一化的值放回字典
                relation_ht_relation[rel][ht] = dict(zip(relation_ht_relation[rel][ht].keys(), normalized_values))
        # norm_fre(relation_ht_relation,1) 
        def get_pr(relation_ht_relation,target_r,relation_heads_tails,triplets,ht=0):
            pr=0
            # if target_r =='/people/cause_of_death/people':
            #     ipdb.set_trace()
            for r in relation_ht_relation: 
                p_r=len(relation_heads_tails[r])/len(triplets)
                if target_r in relation_ht_relation[r][ht]:
                    pr+=p_r*relation_ht_relation[r][ht][target_r]
            return pr
        def get_bayes(relation_ht_relation,relation_heads_tails,triplets,ht=0):
            newrelation_ht_relation = defaultdict(lambda: defaultdict(float))           
            for r in relation_ht_relation: 
                #计算r占比例的三元组
                p_r=len(relation_heads_tails[r])/len(triplets)
                for rel in relation_ht_relation[r][ht]:
                    # if r =='/people/person/profession' and rel =='/people/cause_of_death/people':
                    #     ipdb.set_trace()
                    p_r_r=relation_ht_relation[r][ht][rel]
                    pr=get_pr(relation_ht_relation,rel,relation_heads_tails,triplets,ht)   
                    newrelation_ht_relation[r][rel]= p_r*p_r_r/pr 
            return newrelation_ht_relation
        def second_largest_value(dictionary):
            values = list(dictionary.values())
            values.sort()
            return values[-2] if len(values) > 1 else values[-1]

        # 使用方法
        # newrelation_ht_relation = relation_ht_relation
        #head entity
        newrelation_ht_relation = defaultdict(list) 
        headrhr=get_bayes(relation_ht_relation,relation_heads_tails,triplets,0)  
        #tail entity
        tailrtr=get_bayes(relation_ht_relation,relation_heads_tails,triplets,1)  
        for headrhr_tm in headrhr:
            newrelation_ht_relation[headrhr_tm].append(headrhr[headrhr_tm])
            newrelation_ht_relation[headrhr_tm].append(tailrtr[headrhr_tm])
        norm_fre(newrelation_ht_relation,0) 
        norm_fre(newrelation_ht_relation,1) 
        # ipdb.set_trace()
            
        return newrelation_ht_relation, many_to_one, one_to_many,relation_types,head_relations,relation_heads_tails,relation_tails_heads
    
    relation_ht_relation,many_to_one,one_to_many,relation_types,head_relations,relation_heads_tails,relation_tails_heads=get_rel(biG,triples['train']['pos'])
    # if set_type=='test':
    #     relation_ht_relation,many_to_one,one_to_many,relation_types,head_relations,relation_heads_tails,relation_tails_heads=get_rel(biG,triples['train']['pos']+triples['train']['indpos'])
    # else:
    #     relation_ht_relation,many_to_one,one_to_many,relation_types,head_relations,relation_heads_tails,relation_tails_heads=get_rel(biG,triples['train']['pos'])
           
    def findPaths(G,u,n):
        # paths=[]
        if n==0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        return paths   
    # for i in biG.nodes():
    #     findPaths(biG,i,4)         
    # ipdb.set_trace()  
    flag_on=' '
    path_len_count=[]
    # for i in trange(num_pos_examples):
    #     e1_pos, r_pos, e2_pos = triples[set_type]['pos'][i]
    #     if set_type=='train':
    #         for k,r_dict in biG[e1_pos][e2_pos].items():
    #             try:
    #                 if r_dict:
    #                     if  r_dict['relation'] == r_pos:
    #                         rid=k
    #             except:
    #                 print(biG[e1_pos][e2_pos])
    #                 ipdb.set_trace()
    #         biG.remove_edge(e1_pos,e2_pos,rid)
    #     try:
    #         num=nx.algorithms.shortest_path_length(biG,source=e1_pos, target=e2_pos)
    #     except:
    #         num=0
    #     # if num>4:
    #     #     relations = []
    #     #     path=nx.algorithms.shortest_path(biG,source=e1_pos, target=e2_pos)
    #     #     for i in range(len(path) - 1):
    #     #         edge_data = biG.get_edge_data(path[i], path[i+1])
    #     #         if edge_data is not None:
    #     #             relations.append(edge_data.get(0)['relation'])
    #     #     allentity=[]
    #     #     for i in path:
    #     #         allentity.append(entity2text[i])
    #     #     print(entity2text[e1_pos]+'\t'+r_pos+'\t'+entity2text[e2_pos])
    #     #     print(allentity)
    #     #     print(relations)
    #         # ipdb.set_trace()
    #     path_len_count.append(num)
    #     if set_type=='train':
    #         biG.add_edge(e1_pos,e2_pos,rid)
    # with open(params.dataset+set_type+'path_lencount.txt','w') as f:
    #     for item in path_len_count:
    #         f.write(str(item)+'\n')
    # return
    # for i in trange(num_pos_examples):
        
    #     e1_pos, r_pos, e2_pos = triples[set_type]['pos'][i]
    #     paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_pos, target=e2_pos, cutoff=4)]
    #     valid_paths=paths
    #     if e2_pos in biG[e1_pos]:
    #         r_pos_inds = [k for k, r_dict in
    #                         biG[e1_pos][e2_pos].items() if r_dict['relation'] == r_pos]
    #         if r_pos_inds:
    #             r_pos_inds=r_pos_inds[0]
    #         valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_inds), paths)
    #     lencount=[len(path) for path in valid_paths]
    #     lencount=sorted(lencount)
    #     path_len_count.append(lencount)
        # return
        # ipdb.set_trace()
    for i in trange(num_pos_examples):
        # ipdb.set_trace()
        e1_pos, r_pos, e2_pos = triples[set_type]['pos'][i]
        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_pos, target=e2_pos, cutoff=params.hop)]
       # ipdb.set_trace()
        
        if set_type == 'train':
            assert len(paths) > 0  # at least have (e1, r, e2) in train graph

        if set_type == 'train':
            r_pos_inds = [k for k, r_dict in
                          biG[e1_pos][e2_pos].items() if r_dict['relation'] == r_pos]  # r_ind in biG local graph
            # if len(r_pos_inds) is not 1:
            #     ipdb.set_trace()
            assert len(r_pos_inds) == 1
            r_pos_ind = r_pos_inds[0]
            valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)
        else:
            r_pos_ind = None
            valid_paths = paths

        if valid_paths:
            valid_paths_cnter['valid_paths'] += 1
                
        # [('/m/0hvvf', '/m/039bp', 0)]
        # [('/m/0hvvf', '/m/07s9rl0', 0), ('/m/07s9rl0', '/m/0m9p3', 0), ('/m/0m9p3', '/m/039bp', 0)]
        subgraph_entities_pos = flatten([edge[:2] for edge in flatten(valid_paths)])
        subgraph_entities_pos = list(set(subgraph_entities_pos))
        
        local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
        e1_text_pos, e2_text_pos = local_entity2text[e1_pos], local_entity2text[e2_pos]
        r_text_pos = relation2text[r_pos]
        
        
        # conclusion_pos = question_pattern.format(en2text[e1_pos],en2text[e2_pos],e1_text_pos, r_text_pos, e2_text_pos)
        conclusion_pos = question_pattern.format(e1_text_pos, r_text_pos, e2_text_pos)
        
            
        if params.subgraph_input:
            G_text_pos_edges = construct_subgraph_text(
                biG, subgraph_entities_pos, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
            G_text_pos_edges = ['  '.join(G_text_pos_edges)]
        else:
            # now use path text
            G_text_pos_edges = construct_paths_text(biG, valid_paths, local_entity2text, relation2text, conclusion_relation=r_pos, params=params)
        #增加
        # subgraphtext=generate_subgraph(biG,e1_pos,e2_pos,r_pos,set_type,r_pos_ind,entity2text,relation2text)
        subgraphflag=1
        # if not G_text_pos_edges:
        if 1:
            subgraphflag=0
            # print(1)
            # G_text_pos_edges = construct_subgraph_text(
            #     biG, subgraph_entities_pos, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
            # print(G_text_pos_edges)
            neighbornum=1
            e1_neigh_to_dis = nx.single_source_shortest_path_length(biG, e1_pos, cutoff=neighbornum) # 3 
            e2_neigh_to_dis = nx.single_source_shortest_path_length(biG, e2_pos, cutoff=neighbornum)
            e1_neigh=[i for i in e1_neigh_to_dis.keys() if i is not e1_pos]
            e2_neigh=[i for i in e2_neigh_to_dis.keys() if i is not e2_pos]
            # e1_neigh=random.sample(e1_neigh, min(len(e1_neigh), 3)) # neg = 10
            # e2_neigh=random.sample(e2_neigh, min(len(e2_neigh), 3)) # neg = 10
            
            allsubpath=[]
            e1path=[]
            e2path=[]
            
            for tmp in e1_neigh:
                paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_pos, target=tmp, cutoff=neighbornum)]
                if set_type == 'train':
                    valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)
                else:
                    valid_paths=paths
                if valid_paths:
                    allsubpath.append(valid_paths[0])
                    e1path.append(valid_paths[0])
            for tmp in e2_neigh:
                paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e2_pos, target=tmp, cutoff=neighbornum)]
                if set_type == 'train':
                    valid_paths = get_valid_paths((e2_pos, e1_pos, r_pos_ind), paths)
                else:
                    valid_paths=paths
                if valid_paths:
                    allsubpath.append(valid_paths[0])   
                    e2path.append(valid_paths[0])
            
            subgraph_entities_pos = flatten([edge[:2] for edge in flatten(allsubpath)])
            subgraph_entities_pos = list(set(subgraph_entities_pos))
            
            e1rel=[biG[edge[0]][edge[1]][edge[2]]['relation'] for edge in flatten(e1path)]
            e1rel=[edge[4:] if edge.startswith('inv-') else edge for edge in e1rel]
            e2rel=[biG[edge[0]][edge[1]][edge[2]]['relation'] for edge in flatten(e2path)]
            e2rel=[edge[4:] if edge.startswith('inv-') else edge for edge in e2rel]
            commonrel=set(e1rel)&set(e2rel)
            local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
            
            if not G_text_pos_edges:
                G_text_pos_edges=construct_rel_text(biG,relation_ht_relation,head_relations,
                                                    relation_heads_tails,relation_tails_heads,local_entity2text,relation2text,e1_pos,e2_pos,r_pos,
                                                    one_to_many,many_to_one)
                if r_pos !='/people/person/profession':
                    visual_nei.write(local_entity2text[e1_pos]+' '+relation2text[r_pos]+' '+local_entity2text[e2_pos]+'\t'+G_text_pos_edges[0]+'\n')
                #count_valid[0]: validpos 1:invalidpos 2:validneg 3:invalidneg
                if G_text_pos_edges and G_text_pos_edges[0] is not flag_on:
                    # ipdb.set_trace()
                    count_valid[0]+=1
                else:
                    # ipdb.set_trace()
                    count_valid[1]+=1
                G_text_pos_edges = ['  '.join(G_text_pos_edges)]
                context_num=len(G_text_pos_edges[0].split('; '))
                context_text= G_text_pos_edges
                pos_support.append(len(G_text_pos_edges[0].split(';')))
            else:
                # context_text=construct_rel_text(biG,e1path,e2path,commonrel,local_entity2text,relation2text)
                # context_text=construct_rel_text(biG,relation_ht_relation,head_relations,relation_heads_tails,relation_tails_heads,local_entity2text,relation2text,e1_pos,e2_pos,r_pos)
                count_valid[-2]+=1
                context_text=[' ']
                context_text= ['  '.join(context_text)]
                context_num=len(context_text[0].split('; '))  
                
            if set_type=='train' and not G_text_pos_edges[0]:
                ipdb.set_trace()
                G_text_pos_edges=[' ']     
            
        #增加
            
        def shuffle_G_text_edges(G_text_edges):
            shuffled_G_text = []
            for G_text in G_text_edges:
                G_edges = G_text.strip(';').split('; ')
                random.shuffle(G_edges)
                shuffled_G_text.append('; '.join(G_edges) + ';')
            return shuffled_G_text

        if params.shuffle_times > 0:
            G_text_pos_edges = shuffle_G_text_edges(G_text_pos_edges)

        if G_text_pos_edges and params.combine_paths:
            G_text_pos_edges = [' [SEP] '.join(G_text_pos_edges[:params.kept_paths])]

        # sample/take positive path for train/test
        # if  len(G_text_pos_edges)==0:
        #     ipdb.set_trace() 
        # if context_text is not G_text_pos_edges: 
        #     G_text_pos_edges.append(context_text[0])
            
        for ii, G_text_pos in enumerate(G_text_pos_edges):
            # if context_text is not G_text_pos_edges: 
            #     #  G_text_pos_edges.append(context_text[0])
            #     context_pos = 'Context: {} {} {} {}'.format(G_text_pos+' [SEP] '+context_text[0],en2text[e1_pos],'[SEP]',en2text[e2_pos])
            # else:
            #     context_pos = 'Context: {} {} {} {}'.format(G_text_pos+' [SEP]',en2text[e1_pos],'[SEP]',en2text[e2_pos])

            context_pos = 'Context: {} {} {} {}'.format(G_text_pos+' [SEP]',en2text[e1_pos],'[SEP]',en2text[e2_pos])
                
            # context_pos = 'Context: {} {} {}'.format(G_text_pos,en2text[e1_pos],en2text[e2_pos])
            # context_pos = 'Context: {}'.format(G_text_pos)
            
            if params.block_body:
                context_pos = ''

            if ii >= params.kept_paths and set_type == 'train':  # drop some paths in training
                break

            if G_text_pos:
                
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    1, set_type+'-pos-'+str(i), 'train-pos-'+str(i), conclusion_pos, context_pos))
                foutcontext.write('{}\t{}\t{}\t{}\n'.format(
                    f'{e1_text_pos} {r_text_pos} {e2_text_pos}', context_text[0],context_num,relation2id[r_pos]))  
                valid_paths_cnter[1] += 1

        if set_type == 'train':
            # sampling negative pairs for train
            # this always negative sampling from the neighbors
            pairs = []
            E1_neigh_to_dis = nx.single_source_shortest_path_length(biG, e1_pos, cutoff=3)  # 3 
            E2_neigh_to_dis = nx.single_source_shortest_path_length(biG, e2_pos, cutoff=3)

            common_neighs = set(E1_neigh_to_dis) & set(E2_neigh_to_dis)

            e1_neigh_to_dis = {k: v for k, v in E1_neigh_to_dis.items() if k in common_neighs}
            e2_neigh_to_dis = {k: v for k, v in E2_neigh_to_dis.items() if k in common_neighs}
            
            for neigh in e1_neigh_to_dis:
                if (e1_pos, r_pos, neigh) not in seen_neg and neigh not in (e1_pos, e2_pos):
                    # exclude the sampled negative in training positive
                    # if neigh in G[e1_pos] and r_pos in [r_dict['relation'] for r_dict in G[e1_pos][neigh].values()]:  # (e1_pos, r_pos, neigh) in G train
                    if is_in_G(e1_pos, r_pos, neigh, biG): # (e1_pos, r_pos, neigh) in G train
                        valid_paths_cnter['in_train'] += 1
                        continue

                    pairs.append((e1_pos, neigh))
                    seen_neg.add((e1_pos, r_pos, neigh))
            for neigh in e2_neigh_to_dis:
                if (neigh, r_pos, e2_pos) not in seen_neg and neigh not in (e1_pos, e2_pos):
                    
                    # if e2_pos in G[neigh] and r_pos in [r_dict['relation'] for r_dict in G[neigh][e2_pos].values()]:  # (neigh, r_pos, e2_pos) in G train
                    if is_in_G(neigh, r_pos, e2_pos, biG): # (neigh, r_pos, e2_pos) in G train
                        valid_paths_cnter['in_train'] += 1
                        continue
                    pairs.append((neigh, e2_pos))
                    seen_neg.add((neigh, r_pos, e2_pos))

            # make sure there is a path 3 length path
            if params.dataset.startswith('WN18RR') or  params.dataset.startswith('1fb'):
                pairs=[]
                neg_tails = set(entitylist) - set(biG[e2_pos])
                neg_tails = random.sample(neg_tails, params.neg//2)
                for n in neg_tails:
                    pairs.append((n,e2_pos))
                neg_tails = set(entitylist) - set(biG[e1_pos])
                neg_tails = random.sample(neg_tails, params.neg//2)
                for n in neg_tails:
                    pairs.append((e1_pos,n))

            if G_text_pos_edges:
                pairs = random.sample(pairs, min(len(pairs), params.neg)) # neg = 10
            else:
                pairs = random.sample(pairs, min(len(pairs), 1))
                ipdb.set_trace()
            
            add_neg_num=0
            add_neg_tail=[]
            add_neg_head=[]
            neg_head=list(relation_heads_tails[r_pos].keys())
            neg_head.remove(e1_pos)
            for neg_item in neg_head:
                for t_seg in relation_heads_tails[r_pos][neg_item]:
                    if is_in_G(e1_pos, r_pos, t_seg, biG): 
                        continue
                    add_neg_tail.append((e1_pos,t_seg))
            
            neg_tail=list(relation_tails_heads[r_pos].keys())
            neg_tail.remove(e2_pos) 
            for neg_item in neg_tail:
                for h_seg in relation_tails_heads[r_pos][neg_item]:
                    if is_in_G(h_seg, r_pos, e2_pos, biG): 
                        continue
                    add_neg_head.append((h_seg, e2_pos)) 
            add_neg_tail=random.sample(add_neg_tail, min(len(add_neg_tail), add_neg_num))  
            add_neg_head=random.sample(add_neg_head, min(len(add_neg_head), add_neg_num))  
            pairs=pairs+add_neg_tail+add_neg_head
            # if len(pairs)<params.neg and params.dataset.startswith('WN18RR'):
            # if len(pairs)<params.neg:
            #     # ipdb.set_trace()
            #     completenum=params.neg-len(pairs)
            #     # 需要剔除已经放入负采样中的和真实三元组
            #     allneglist=list(set(entitylist)-set([e1_pos,e2_pos])-set(common_neighs))
                
            #     negnodes=random.sample(allneglist,completenum)
            #     for ii in range(completenum):
                    
            #         if ii < completenum//2:
            #             if not biG.has_edge(negnodes[ii],e2_pos):
            #                 pairs.append((negnodes[ii],e2_pos))
            #         else:
            #             if not biG.has_edge(e1_pos,negnodes[ii]):
            #                 pairs.append((e1_pos,negnodes[ii]))



            for j, (e1_neg, e2_neg) in enumerate(pairs):
                paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_neg, target=e2_neg, cutoff=params.hop)]
               
                valid_paths = get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)  # in train remove all path contains e1_pos, r_pos, e2_pos,
                # why also do this for negative examples?
                assert biG[e1_pos][e2_pos][r_pos_ind]['relation'] == r_pos
             
                subgraph_entities_neg = flatten([edge[:2] for edge in flatten(valid_paths)])
                subgraph_entities_neg = list(set(subgraph_entities_neg))

                local_entity2text = construct_local_entity2text(subgraph_entities_neg, entity2text)
                e1_text_neg, e2_text_neg = local_entity2text[e1_neg], local_entity2text[e2_neg]
                r_text_neg = r_text_pos
             #增加
                
                #增加
                if params.subgraph_input:
                    G_text_neg_edges = construct_subgraph_text(
                        G, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
                    G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                else:
                    G_text_neg_edges = construct_paths_text(biG, valid_paths, local_entity2text, relation2text, conclusion_relation=r_pos, params=params)
        #增加   
                # subgraphtext=generate_subgraph(biG,e1_neg,e2_neg,r_pos,set_type,r_pos_ind,entity2text,relation2text)
        
                subgraphflag=1
                # if not G_text_neg_edges:
                if 1:
                    subgraphflag=0
                    neighbornum=1
                    e1_neigh_to_dis = nx.single_source_shortest_path_length(biG, e1_neg, cutoff=neighbornum) # 3 
                    e2_neigh_to_dis = nx.single_source_shortest_path_length(biG, e2_neg, cutoff=neighbornum)
                    e1_neigh=[i for i in e1_neigh_to_dis.keys() if i is not e1_neg]
                    e2_neigh=[i for i in e2_neigh_to_dis.keys() if i is not e2_neg]
                    # e1_neigh=random.sample(e1_neigh, min(len(e1_neigh), 3)) # neg = 10
                    # e2_neigh=random.sample(e2_neigh, min(len(e2_neigh), 3)) # neg = 10
                    e1path=[]
                    e2path=[]
                    allsubpath=[]
                    # subgraph_entities_pos = flatten([edge[:2] for edge in flatten(valid_paths)])
                    # subgraph_entities_pos = list(set(subgraph_entities_pos))
                    # local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
                    
                    for tmp in e1_neigh:
                        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_neg, target=tmp, cutoff=neighbornum)]                   
                        valid_paths=get_valid_paths((e1_pos, e2_pos, r_pos_ind), paths)
                        if valid_paths:
                            allsubpath.append(valid_paths[0])
                            e1path.append(valid_paths[0])
                    for tmp in e2_neigh:
                        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e2_neg, target=tmp, cutoff=neighbornum)]
                        valid_paths=get_valid_paths((e2_pos, e1_pos, r_pos_ind), paths)
                        
                        if valid_paths:
                            allsubpath.append(valid_paths[0])   
                            e2path.append(valid_paths[0])
                            
                    if  len(allsubpath)==0:
                        continue
                    e1rel=[biG[edge[0]][edge[1]][edge[2]]['relation'] for edge in flatten(e1path)]
                    e1rel=[edge[4:] if edge.startswith('inv-') else edge for edge in e1rel]
                    e2rel=[biG[edge[0]][edge[1]][edge[2]]['relation'] for edge in flatten(e2path)]
                    e2rel=[edge[4:] if edge.startswith('inv-') else edge for edge in e2rel]
                    commonrel=set(e1rel)&set(e2rel)
                    subgraph_entities_neg = flatten([edge[:2] for edge in flatten(allsubpath)])
                    subgraph_entities_neg = list(set(subgraph_entities_neg))
                    local_entity2text = construct_local_entity2text(subgraph_entities_neg, entity2text)
                    if not G_text_neg_edges:
                        G_text_neg_edges=construct_rel_text(biG,relation_ht_relation,head_relations,relation_heads_tails,
                                                            relation_tails_heads,local_entity2text,relation2text,e1_neg,e2_neg,r_pos,
                                                            one_to_many,many_to_one)
                        
                        # G_text_neg_edges=[' ']
                        if  G_text_neg_edges and G_text_neg_edges[0] is not flag_on:
                            count_valid[2]+=1
                        else:
                            count_valid[3]+=1
                        # G_text_neg_edges=construct_rel_text(biG,e1path,e2path,commonrel,local_entity2text,relation2text)
                        G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                        context_num=len(G_text_neg_edges[0].split('; '))
                        context_text=G_text_neg_edges
                        neg_support.append(len(G_text_neg_edges[0].split(';')))
                    else:
                        count_valid[-1]+=1
                        
                        context_text=[' ']
                        # context_text=construct_rel_text(biG,relation_ht_relation,head_relations,relation_heads_tails,
                        #                                     relation_tails_heads,local_entity2text,relation2text,e1_neg,e2_neg,r_pos)
                        context_text = ['  '.join(context_text)]
                        context_num=len(context_text[0].split('; '))
                        
                   
                    
               
        #增加


                if params.shuffle_times > 0:
                    G_text_neg_edges = shuffle_G_text_edges(G_text_neg_edges)

                if G_text_neg_edges and params.combine_paths:
                    G_text_neg_edges = [' [SEP] '.join(G_text_neg_edges[:params.kept_paths])]
                # if G_text_neg_edges[0]=='':
                #     ipdb.set_trace()
                # if context_text is not G_text_neg_edges: 
                #     G_text_neg_edges.append(context_text[0])
            
                for jj, G_text_neg in enumerate(G_text_neg_edges):
                    
                    if jj >= params.kept_paths and set_type == 'train':  # drop some paths in training for neg
                        break
                    # conclusion_neg = question_pattern.format(en2text[e1_neg],en2text[e2_neg],e1_text_neg, r_text_neg, e2_text_neg)
                    conclusion_neg = question_pattern.format(e1_text_neg, r_text_neg, e2_text_neg)
                    
                    # if context_text is not G_text_neg_edges:
                    #     # ipdb.set_trace()  
                    #     context_neg = 'Context: {} {} {} {}'.format(G_text_neg+' [SEP] '+context_text[0],en2text[e1_neg],'[SEP]',en2text[e2_neg])
                    # else:
                    #     context_neg = 'Context: {} {} {} {}'.format(G_text_neg+' [SEP]',en2text[e1_neg],'[SEP]',en2text[e2_neg])
                    context_neg = 'Context: {} {} {} {}'.format(G_text_neg+' [SEP]',en2text[e1_neg],'[SEP]',en2text[e2_neg])
                    
                    # context_neg = 'Context: {} {} {}'.format(G_text_neg,en2text[e1_neg],en2text[e2_neg])
                    # context_neg = 'Context: {}'.format(G_text_neg)
                    
                    if params.block_body:
                        context_neg = ''
                    # context_neg, conclusion_neg = conclusion_neg, context_neg
                    if G_text_neg:
                        fout.write('{}\t{}\t{}\t{}\t{}\n'.format(0, set_type+'-neg-'+str(i)+'-'+str(
                            j), set_type+'-neg-'+str(i)+'-'+str(j), conclusion_neg, context_neg))
                        foutcontext.write('{}\t{}\t{}\t{}\n'.format(
                            f'{e1_text_neg} {r_text_neg} {e2_text_neg}', context_text[0],context_num,relation2id[r_pos]))  
                        valid_paths_cnter[0] += 1

        elif set_type == 'test' or set_type == 'dev':
            # take pre-generated ranking head/tail triplets
            num_empty_path_neg_this_pos = 0
            for j in range(num_neg_samples_per_link):  # pos i 's jth neg
                """
                    pbar = tqdm(total=len(pos_edges))
                    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
                        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(
                            pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
                    pos1' neg, pos2's neg, ...
                """
                e1_neg, r_neg, e2_neg = triples[set_type]['neg'][i * num_neg_samples_per_link + j]

                paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_neg, target=e2_neg, cutoff=params.hop)]  # be careful to choose biG
                
                # subgraphtext=generate_subgraph(biG,e1_neg,e2_neg,r_pos,set_type,r_pos_ind,entity2text,relation2text)
                subgraphflag=1
                valid_paths = paths


                subgraph_entities_neg = flatten([edge[:2] for edge in flatten(paths)])
                subgraph_entities_neg = list(set(subgraph_entities_neg))

                local_entity2text = construct_local_entity2text(subgraph_entities_neg, entity2text)
                e1_text_neg, e2_text_neg = local_entity2text[e1_neg], local_entity2text[e2_neg]
                r_text_neg = relation2text[r_neg]

                # conclusion_neg = question_pattern.format(en2text[e1_neg],en2text[e2_neg],e1_text_neg, r_text_neg, e2_text_neg)
                # conclusion_neg = question_pattern.format(e1_text_neg, r_text_neg, e2_text_neg,en2text[e1_neg],en2text[e2_neg])
                conclusion_neg = question_pattern.format(e1_text_neg, r_text_neg, e2_text_neg)

                # G_text_neg_edges = construct_subgraph_text(
                # G, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos), join_edge_text=False)

                if params.subgraph_input:
                    G_text_neg_edges = construct_subgraph_text(
                        biG, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_pos, r_pos, e2_pos))
                    G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                else:
                    # now use path text
                    G_text_neg_edges = construct_paths_text(biG, valid_paths, local_entity2text, relation2text, conclusion_relation=r_neg, params=params)

                # if len(valid_paths) == 0:
                if 1:
                    subgraphflag=0
                    
                    num_empty_path_neg_this_pos += 1
                    #增加
                    neighbornum=1
                    e1_neigh_to_dis = nx.single_source_shortest_path_length(biG, e1_neg, cutoff=neighbornum) # 3 
                    e2_neigh_to_dis = nx.single_source_shortest_path_length(biG, e2_neg, cutoff=neighbornum)
                    e1_neigh=[i for i in e1_neigh_to_dis.keys() if i is not e1_neg]
                    e2_neigh=[i for i in e2_neigh_to_dis.keys() if i is not e2_neg]
                    # e1_neigh=random.sample(e1_neigh, min(len(e1_neigh), 3)) # neg = 10
                    # e2_neigh=random.sample(e2_neigh, min(len(e2_neigh), 3)) # neg = 10
                    
                    allsubpath=[]
                    e1path=[]
                    e2path=[]
                    # subgraph_entities_pos = flatten([edge[:2] for edge in flatten(valid_paths)])
                    # subgraph_entities_pos = list(set(subgraph_entities_pos))
                    # local_entity2text = construct_local_entity2text(subgraph_entities_pos, entity2text)
                    
                    for tmp in e1_neigh:
                        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e1_neg, target=tmp, cutoff=neighbornum)]                   
                        valid_paths=paths
                        if valid_paths:
                            allsubpath.append(valid_paths[0])
                            e1path.append(valid_paths[0])
                    for tmp in e2_neigh:
                        #筛选一下含有多个种类relation的path两个实体之间存在多种relation 不只是path[0]
                        paths = [*nx.algorithms.all_simple_edge_paths(biG, source=e2_neg, target=tmp, cutoff=neighbornum)]
                        valid_paths=paths
                        if valid_paths:
                            allsubpath.append(valid_paths[0])  
                            e2path.append(valid_paths[0])
                             
                    if  len(allsubpath)==0:
                        continue
                    e1rel=[biG[edge[0]][edge[1]][edge[2]]['relation'] for edge in flatten(e1path)]
                
                    e1rel=[edge[4:] if edge.startswith('inv-') else edge for edge in e1rel]
                    e2rel=[biG[edge[0]][edge[1]][edge[2]]['relation'] for edge in flatten(e2path)]
                    e2rel=[edge[4:] if edge.startswith('inv-') else edge for edge in e2rel]
                    commonrel=set(e1rel)&set(e2rel)
                    subgraph_entities_neg = flatten([edge[:2] for edge in flatten(allsubpath)])
                    subgraph_entities_neg = list(set(subgraph_entities_neg))
                    local_entity2text = construct_local_entity2text(subgraph_entities_neg, entity2text)
                    # ipdb.set_trace()
                    if not G_text_neg_edges:
                        # if not commonrel:                      
                        
                        #     G_text_neg_edges = construct_subgraph_text(
                        #         biG, subgraph_entities_neg, local_entity2text, relation2text, excluded=(e1_neg, r_neg, e2_neg))
                        #     G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                        #     G_text_neg_edges=[]
                        # else:   
                        #     G_text_neg_edges=construct_commonrel_text(biG,e1path,e2path,commonrel,local_entity2text,relation2text)
                        #     G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                        # G_text_neg_edges=construct_rel_text(biG,e1path,e2path,commonrel,local_entity2text,relation2text)
                        G_text_neg_edges=construct_rel_text(biG,relation_ht_relation,head_relations,relation_heads_tails,
                                                            relation_tails_heads,local_entity2text,relation2text,e1_neg,e2_neg,r_neg,
                                                            one_to_many,many_to_one)
                        
                        if  G_text_neg_edges and G_text_neg_edges[0] is not flag_on:
                            count_valid[2]+=1
                        else:
                            count_valid[3]+=1
                        G_text_neg_edges = ['  '.join(G_text_neg_edges)]
                        context_num=len(G_text_neg_edges[0].split('; '))
                        context_text=G_text_neg_edges
                        neg_support.append(len(G_text_neg_edges[0].split(';')))
                        
                    else:
                        count_valid[-1]+=1
                        context_text=[' ']
                        # context_text=construct_rel_text(biG,e1path,e2path,commonrel,local_entity2text,relation2text)
                        context_text = ['  '.join(context_text)]
                        context_num=len(context_text[0].split('; '))
                    
                   
                   
                    

                if params.shuffle_times > 0:
                    G_text_neg_edges = shuffle_G_text_edges(G_text_neg_edges)

                if G_text_neg_edges and params.combine_paths:
                    G_text_neg_edges = [' [SEP] '.join(G_text_neg_edges[:params.kept_paths])]
                # if  len(G_text_neg_edges)==0:
                #     ipdb.set_trace()
                # G_text_neg = ' '.join(G_text_neg_edges)
                # if context_text is not G_text_neg_edges: 
                #     G_text_neg_edges.append(context_text[0])
            
                for G_text_neg in G_text_neg_edges:
                    
                    # if context_text is not G_text_neg_edges:
                         
                    #     context_neg = 'Context: {} {} {} {}'.format(G_text_neg+' [SEP] '+context_text[0],en2text[e1_neg],'[SEP]',en2text[e2_neg])
                    # else:
                    #     context_neg = 'Context: {} {} {} {}'.format(G_text_neg+' [SEP]',en2text[e1_neg],'[SEP]',en2text[e2_neg])
                    context_neg = 'Context: {} {} {} {}'.format(G_text_neg+' [SEP]',en2text[e1_neg],'[SEP]',en2text[e2_neg])
                    
                    
                    # context_neg = 'Context: {} {} {}'.format(G_text_neg,en2text[e1_neg],en2text[e2_neg])
                    
                    # context_neg = 'Context: {}'.format(G_text_neg)
                    if params.block_body:
                        context_neg = ''
                    if G_text_neg:
                        fout.write('{}\t{}\t{}\t{}\t{}\n'.format(
                            0, set_type+'-neg-'+str(i)+'-'+str(j), set_type+'-neg-'+str(i)+'-'+str(j), conclusion_neg, context_neg))
                        foutcontext.write('{}\t{}\t{}\t{}\n'.format(
                        f'{e1_text_neg} {r_text_neg} {e2_text_neg}', context_text[0],context_num,relation2id[r_neg]))  
                        valid_paths_cnter[0] += 1

    print('# statistics: ', valid_paths_cnter)
    print(count_valid)
    visual_nei.close()
    foutcontext.close()
    fout.close()
    # ipdb.set_trace()
def load_train(params):
    # construct graph
    triples = {'train': defaultdict(list), 'valid': defaultdict(
        list), 'test': defaultdict(list)}


    # entities=set()
    # handr=defaultdict(list)
    # tandr=defaultdict(list)
    # with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/train.txt') as fin:
    #     for i, l in enumerate(fin):
    #         e1, r, e2 = l.strip().split('\t')
    #         entities.add(e1)
    #         entities.add(e2)
    #         handr[(e1,r)].append(e2)
    #         tandr[(r,e2)].append(e1)
    # with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/test.txt') as fin:
    #     for i, l in enumerate(fin):
    #         e1, r, e2 = l.strip().split('\t')
    #         entities.add(e1)
    #         entities.add(e2)
    #         handr[(e1,r)].append(e2)
    #         tandr[(r,e2)].append(e1)
    # with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/valid.txt') as fin:
    #     for i, l in enumerate(fin):
    #         e1, r, e2 = l.strip().split('\t')
    #         entities.add(e1)
    #         entities.add(e2)
    #         handr[(e1,r)].append(e2)
    #         tandr[(r,e2)].append(e1)        
    # with open('ranking_head.txt','w')as f:   
    #     with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/test.txt') as fin:
    #         for i, l in enumerate(fin):
    #             e1, r, e2 = l.strip().split('\t')
    #             f.write(str(1)+'\t'+e1+'\t'+r+'\t'+e2+'\n')
    #             for ii in entities:
    #                 if ii is not e2 and ii not in handr[(e1,r)] and ii is not e1:
    #                     f.write(str(0)+'\t'+e1+'\t'+r+'\t'+ii+'\n')
    # ipdb.set_trace()                  
    biG = nx.MultiDiGraph()
    # all postive train
    with open(f'{params.main_dir}/data/{params.dataset}/train_{params.part}.txt') as fin:
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples[params.set_type]['pos'].append([e1, r, e2])
            # bidirectional G
            biG.add_edges_from([(e1, e2, dict(relation=r))])
            biG.add_edges_from([(e2, e1, dict(relation='inv-'+r))])


    generate_bert_input_from_scratch(biG, 'train', triples, params=params)


def load_test(params):

    # construct graph
    triples = {'train': defaultdict(list), 'test': defaultdict(list)}
    # ipdb.set_trace()
    biG = nx.MultiDiGraph()
    # all train
    with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/train.txt') as fin:  # fb237_ind, use full inductive test fact graph
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples['train']['pos'].append([e1, r, e2])
            # bidirectional G
            biG.add_edges_from([(e1, e2, dict(relation=r))])
            biG.add_edges_from([(e2, e1, dict(relation='inv-'+r))])
    with open(f'{params.main_dir}/data/{params.dataset}/train.txt') as fin:  # fb237_ind, use full inductive test fact graph
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples['train']['indpos'].append([e1, r, e2])
    # with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/test.txt') as fin:  # fb237_ind, use full inductive test fact graph
    #     for l in fin:
    #         e1, r, e2 = l.strip().split('\t')
    #         triples['test']['pos'].append([e1, r, e2])
            
    # with open(f'./ranking_head.txt') as fin:  # fb237_ind
    #     for i, l in enumerate(fin):
    #         label,e1, r, e2 = l.strip().split('\t')
    #         if int(label) == 1:
    #             triples[params.set_type]['pos'].append([e1, r, e2])
    #         else:
    #             triples[params.set_type]['neg'].append([e1, r, e2])                 
    
    # load from unified bertrl generated test

    num_samples = params.candidates
    for head_type in ['head', 'tail']:
        with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/ranking_{head_type}.txt') as fin:  # fb237_ind
            for i, l in enumerate(fin):
                e1, r, e2 = l.strip().split('\t')
                if i % num_samples == 0:
                    triples[params.set_type]['pos'].append([e1, r, e2])
                else:
                    triples[params.set_type]['neg'].append([e1, r, e2])
    # ipdb.set_trace()
    generate_bert_input_from_scratch(biG, 'test', triples, params=params)


def load_valid(params):

    # construct graph
    triples = {'train': defaultdict(list), 'dev': defaultdict(list)}

    biG = nx.MultiDiGraph()
    # all train
    with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/train.txt') as fin:  # fb237v1_ind, use full inductive test fact graph
        for l in fin:
            e1, r, e2 = l.strip().split('\t')
            triples['train']['pos'].append([e1, r, e2])
            # bidirectional G
            biG.add_edges_from([(e1, e2, dict(relation=r))])
            biG.add_edges_from([(e2, e1, dict(relation='inv-'+r))])

    all_entities = list(biG)
    for head_type in ['head', 'tail']:
        with open(f'{params.main_dir}/data/{params.dataset}{params.ind_suffix}/valid.txt') as fin:  # fb237v1_ind, use full inductive test fact graph
            for l in fin:
                e1_pos, r_pos, e2_pos = l.strip().split('\t')
                triples['dev']['pos'].append([e1_pos, r_pos, e2_pos])  # totally append two times pos

                # sampling negative pairs
                pairs_unreached = []
                if head_type == 'head':
                    j = 0
                    while j < params.candidates:
                        e2_neg = random.choice(all_entities)
                        if not is_in_G(e1_pos, r_pos, e2_neg, biG):
                            triples['dev']['neg'].append([e1_pos, r_pos, e2_neg])
                            j += 1
                else:
                    j = 0
                    while j < params.candidates:
                        e1_neg = random.choice(all_entities)
                        if not is_in_G(e1_neg, r_pos, e2_pos, biG):
                            triples['dev']['neg'].append([e1_neg, r_pos, e2_pos])
                            j += 1

    generate_bert_input_from_scratch(biG, 'dev', triples, params=params)


def main(params):

    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    params.dataset_short = params.dataset.split('_')[0]  # without suffix
    params.bertrl_data_dir = f'{params.main_dir}/bertrl_data/{params.dataset}_hop{params.hop}_{params.part}{params.suffix}'

    if params.dataset.startswith('WN18RR'):
        params.text_data_dir = f'{params.main_dir}/data/text/WN18RR/'
    elif params.dataset.startswith('fb'):
        params.text_data_dir = f'{params.main_dir}/data/text/FB237/'
    elif params.dataset.startswith('nell'):
        params.text_data_dir = f'{params.main_dir}/data/text/NELL995/'
    elif params.dataset.startswith('testgraph'):
        params.text_data_dir = f'{params.main_dir}/data/text/testgraph/'
    else:
        params.text_data_dir = f'{params.main_dir}/data/text/'+params.dataset

    entity2text = {}
    # entity2longtext = {}
    with open(f'{params.text_data_dir}/entity2text.txt') as fin:
        for l in fin:
            entity, text = l.strip().split('\t')
            name = text.split(',')[0]
            entity2text[entity] = name
            # entity2longtext[entity] = text
    relation2text = {}
    with open(f'{params.text_data_dir}/relation2text.txt') as fin:
        for l in fin:
            relation, text = l.strip().split('\t')
            relation2text[relation] = text

    params.entity2text, params.relation2text = entity2text, relation2text
    # params.entity2longtext = entity2longtext

    if params.block_body:
        params.bertrl_data_dir += '_block_body'

    if not os.path.exists(params.bertrl_data_dir):
        os.makedirs(params.bertrl_data_dir)

    if params.set_type == 'train':
        load_train(params)
    elif params.set_type == 'test':
        load_test(params)
    elif params.set_type == 'dev':
        load_valid(params)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='BERTRL model')
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--part", type=str, default="full",
                        help="part")

    # Data processing pipeline params
    parser.add_argument("--hop", type=int, default=3,
                        help="max reasoning path length")
    parser.add_argument('--set_type', '-st', type=str, default='train',
                        help='set type of train/valid/test')
    parser.add_argument("--shuffle_times", type=int, default=0,
                        help="Shuffle times")
    parser.add_argument("--kept_paths", type=int, default=3,
                        help="number of kept sub paths")

    parser.add_argument("--suffix", type=str, default="",
                        help="additional suffix")
    parser.add_argument("--downsample", default=False, action='store_true',
                        help="downsample or not")
    parser.add_argument("--block_body", default=False, action='store_true',
                        help="block body or not")
    parser.add_argument("--ind_suffix", type=str, default='_ind',
                        help="ind suffix")
    parser.add_argument("--use_reversed_order", default=False, action='store_true',
                        help="use reversed order or not")
    parser.add_argument("--sort_by_len", default=False, action='store_true',
                        help="sort_by_len ")
    parser.add_argument("--combine_paths", default=False, action='store_true',
                        help="combine_paths ")
    parser.add_argument("--subgraph_input", default=False, action='store_true',
                        help="subgraph_input ")

    parser.add_argument("--neg", type=int, default=10,
                        help="neg")
    parser.add_argument("--candidates", type=int, default=50,
                        help="number of candidates for ranking")

    params = parser.parse_args()
    main(params)
