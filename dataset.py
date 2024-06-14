import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

import torch
from sklearn.metrics import average_precision_score
from collections import defaultdict, Counter
import itertools
import ipdb
from model import BertForKBCSequenceClassification
import pickle

class ContextDataset(Dataset):
    def __init__(self, args, data_args, tokenizer,mode):
        self.args = args
        self.data_args=data_args
        self.data_name = data_args.data_dir+'/'+mode+'context.tsv'
        # self.data_name = data_args.data_dir+'/'+mode+'.tsv'
        self.data = self.load_dataset()
        self.mode = mode
        self.tokenizer=tokenizer    
        self.gluedata=GlueDataset(data_args, tokenizer=tokenizer,mode=mode)
        
    def load_dataset(self):
        data_name_path = self.data_name
        contents = []

        output_pickle = data_name_path[0: data_name_path.rfind('.')] + '.pkl'
        # if os.path.exists(output_pickle):
        #     with open(output_pickle, 'rb') as handle:
        #         contents = pickle.load(handle)
        #         return contents

        with open(data_name_path, 'r', encoding='UTF-8') as f:
            next(f)
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                tar_triple, context_triple, context_num,rid= [_.strip() for _ in line.split('\t')]   
                # aaa,bbb,ccc,tar_triple, context_triple = [_.strip() for _ in line.split('\t')]             
                # ipdb.set_trace()
                # contents.append((tar_triple, context_triple))
                contents.append(rid)

        # with open(output_pickle, 'wb') as handle:
        #     pickle.dump(contents, handle)

        return contents

    def __getitem__(self, index):
    #     tar_triple = self.data[index][0]
    #     context_triple = self.data[index][1]
        
    # #     # context_triple=context_triple.split('; ')
        
    # #     all_triple=[]
    # #     all_triple.append(tar_triple)
    # #     all_triple.append(context_triple)
        
    # #     # all_triple.extend(context_triple)
        
    #     tar_encoding = self.tokenizer(
    #     tar_triple,
    #     max_length=30,
    #     padding="max_length",
    #     truncation=True,
    # )
        
    #     batch_encoding = self.tokenizer(
    #     context_triple,
    #     max_length=100,
    #     padding="max_length",
    #     truncation=True,
    # )
       
    #     tar_encoding['input_ids'].extend(batch_encoding['input_ids'])
    #     tar_encoding['attention_mask'].extend(batch_encoding['attention_mask'])
    #     return [self.gluedata[index],tar_encoding]
        # ipdb.set_trace()
        
        return [self.gluedata[index],self.data[index]]

        

    def __len__(self):
        # return len(self.gluedata)
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        sample_et_content_list = []
        sample_et_content_list.append([_[0] for _ in batch])

        sample_kg_content_list = []
        sample_kg_content_list.append([_[1] for _ in batch])

        gt_ent_list = []
        gt_ent_list.append([_[2] for _ in batch])

        et_content = torch.LongTensor(sample_et_content_list[0])
        kg_content = torch.LongTensor(sample_kg_content_list[0])

        gt_ent = torch.LongTensor(gt_ent_list[0])

        return et_content, kg_content, gt_ent

