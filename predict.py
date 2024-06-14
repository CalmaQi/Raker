""" Finetuning BERTRL for scoring.
Adapted from huggingface transformers sequence classification on GLUE"""


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
from model import BertForKBCSequenceClassification,SoftEmbedding
from dataset import ContextDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

class myDataset:
    def __init__(self, args, train_dataset):
        self.args = args
        self.train_dataset = train_dataset
    @staticmethod
    def data_collator(batch):
        ntoken=10
        input_ids = to_indices([torch.cat([torch.LongTensor(ex[0].input_ids)[0].unsqueeze(0),torch.full((1, ntoken), 100).resize(ntoken),torch.LongTensor(ex[0].input_ids)[1:]]) for ex in batch])
        attention_mask = to_indices([torch.cat((torch.full((1, ntoken), 1).resize(ntoken),torch.LongTensor(ex[0].attention_mask))) for ex in batch])
        token_type_ids= to_indices([torch.cat((torch.full((1, ntoken), 0).resize(ntoken),torch.LongTensor(ex[0].token_type_ids))) for ex in batch])
        rid=torch.LongTensor([int(ex[1]) for ex in batch])
        
        # input_ids = to_indices([torch.LongTensor(ex[0].input_ids) for ex in batch])
        # attention_mask = to_indices([torch.LongTensor(ex[0].attention_mask) for ex in batch])
        # token_type_ids= to_indices([torch.LongTensor(ex[0].token_type_ids) for ex in batch])
        #batch_size,context_num,max_len
        # ipdb.set_trace()
        # context_input_ids=context_indices([torch.cat((torch.full((1, ntoken), 500).resize(ntoken),torch.LongTensor(ex[1]['input_ids']))) for ex in batch])
        # context_attention_mask=context_indices([torch.cat((torch.full((1, ntoken), 1).resize(ntoken),torch.LongTensor(ex[1]['attention_mask']))) for ex in batch])
        # context_input_ids=context_indices([torch.LongTensor(ex[1]['input_ids']) for ex in batch])
        # context_attention_mask=context_indices([torch.LongTensor(ex[1]['attention_mask']) for ex in batch])
        # context_token_type_ids=context_indices([torch.LongTensor(ex[1]['token_type_ids']) for ex in batch])
        
        # context_input_ids=context_indices([torch.LongTensor(ex[1]['input_ids']) for ex in batch])
        # context_attention_mask=context_indices([torch.LongTensor(ex[1]['attention_mask']) for ex in batch])
        # # context_token_type_ids=context_indices([torch.LongTensor(ex[1]['token_type_ids']) for ex in batch])
        
       
        # # context_text={'input_ids':context_input_ids,'attention_mask':context_attention_mask,'token_type_ids':context_token_type_ids}
        # context_text={'input_ids':context_input_ids,'attention_mask':context_attention_mask}
        if batch[0][0].label is not None :
            labels= torch.LongTensor([(ex[0].label) for ex in batch])
            # return{'input_ids':None,
            #    'attention_mask':None,
            #    'token_type_ids':None,
            #    'labels':labels,
            #    'context_text':context_text
            #    }
            return{
                 'input_ids':input_ids,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids,
               'rs_tensor':rid,
                
               'labels':labels,
            # 'context_text':context_text
            
               }
        else:
            return{
                'input_ids':input_ids,
                'attention_mask':attention_mask,
                'token_type_ids':token_type_ids,
               'rs_tensor':rid,
            # 'context_text':context_text
               }      
        # ntoken=10
        # # input_ids = to_indices([torch.cat((torch.full((1, ntoken), 500).resize(ntoken),torch.LongTensor(ex.input_ids))) for ex in batch])
        # # attention_mask = to_indices([torch.cat((torch.full((1, ntoken), 1).resize(ntoken),torch.LongTensor(ex.attention_mask))) for ex in batch])
        # # token_type_ids= to_indices([torch.cat((torch.full((1, ntoken), 0).resize(ntoken),torch.LongTensor(ex.token_type_ids))) for ex in batch])
        
        # input_ids = to_indices([torch.LongTensor(ex[0].input_ids) for ex in batch])
        # attention_mask = to_indices([torch.LongTensor(ex[0].attention_mask) for ex in batch])
        # token_type_ids= to_indices([torch.LongTensor(ex[0].token_type_ids) for ex in batch])
        # #batch_size,context_num,max_len
        # # ipdb.set_trace()
        # context_input_ids=context_indices([torch.cat((torch.full((1, ntoken), 500).resize(ntoken),torch.LongTensor(ex[1]['input_ids']))) for ex in batch])
        # context_attention_mask=context_indices([torch.cat((torch.full((1, ntoken), 1).resize(ntoken),torch.LongTensor(ex[1]['attention_mask']))) for ex in batch])
        # # context_input_ids=context_indices([torch.LongTensor(ex[1]['input_ids']) for ex in batch])
        # # context_attention_mask=context_indices([torch.LongTensor(ex[1]['attention_mask']) for ex in batch])
        # # context_token_type_ids=context_indices([torch.LongTensor(ex[1]['token_type_ids']) for ex in batch])
        
        # context_text={'input_ids':context_input_ids,'attention_mask':context_attention_mask}
        # if batch[0][0].label is not None :
        #     labels= torch.LongTensor([(ex[0].label) for ex in batch])
        #     return{'input_ids':input_ids,
        #        'attention_mask':attention_mask,
        #        'token_type_ids':token_type_ids,
        #        'labels':labels,
        #        'context_text':context_text
        #        }
        # else:
        #     return{'input_ids':input_ids,
        #        'attention_mask':attention_mask,
        #        'token_type_ids':token_type_ids,
        #        'context_text':context_text
        #        }    
def context_indices(batch_tensor):
    return torch.cat([i.unsqueeze(0) for i in batch_tensor])

    batch_size=len(batch_tensor)
    contextnum,max_len = batch_tensor[0].shape
    # ipdb.set_trace()
    indices = torch.LongTensor(batch_size, contextnum,max_len)
    # For BERT, mask value of 1 corresponds to a valid position
    for i, t in enumerate(batch_tensor):
        indices[i,:len(t)].copy_(t)
        # indices[i, :len(t)]=(t)
        
    return indices        
def to_indices(batch_tensor):
    batch_size=len(batch_tensor)
    max_len = len(batch_tensor[0])
    indices = torch.LongTensor(batch_size, max_len)
    # For BERT, mask value of 1 corresponds to a valid position
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        # indices[i, :len(t)]=(t)
        
    return indices

def compute_ranking_metrics_fn(predictions,data_args):
    ipdb.set_trace()
    t_res = torch.softmax(torch.from_numpy(predictions), 1)
    eid2pos = {}
    eid2neg = defaultdict(list)

    bertrl_data_file = f'{data_args.data_dir}/test.tsv' # bertrl data file (linearized)
    examples = [l.strip().split('\t') for l in open(bertrl_data_file)][1:]

    # another set of evaluation
    eid2pos = defaultdict(list)
    eid2neg = defaultdict(list)
    eid2neg_lv2 = {}
    for i, example in enumerate(examples):
        label = int(example[0])
        eid = example[1].split('-')[2] # test-neg-2-24
        if label == 1:
            eid2pos[eid].append(i)
        else:
            e_negid = example[1].split('-')[3] # test-neg-2-24
            eid2neg[eid].append(i)
            if eid not in eid2neg_lv2:
                eid2neg_lv2[eid] = defaultdict(list)
            eid2neg_lv2[eid][e_negid].append(i)

    
    hit1 = 0 
    # hits = {1:0, 2:0, 3:0, 4:0, 5:0, 10:0, 20:0, 30:0, 40:0, 50:0}
    hits = Counter()
    ranks = []
    for eid, pos_is in eid2pos.items():
        pos_scores = t_res[pos_is, 1]
        # posscore=t_res[eid2pos['32'],1]
        # eid2neg_lv2['186']
        # negindex=[ ii[0] for ii in eid2neg_lv2['186'].values()]
        # negscore=t_res[negindex,1]
        neg_is = eid2neg[eid]
        geq_j = 1  # rank
        if not neg_is:  # no negative, only positive
            hit1 += 1
        else:
            neg_scores_of_eid = []
            neg_scores_lists = []
            for neg_eid in eid2neg_lv2[eid]:
                neg_is_of_eid = eid2neg_lv2[eid][neg_eid]
                neg_scores_ = t_res[neg_is_of_eid, 1]  # previously a bug here, previously use neg_is which lowers the hit 2+ performance
                neg_max_score_ = torch.max(neg_scores_).item()
                neg_scores_of_eid.append(neg_max_score_)
                neg_scores_lists.append(neg_scores_.sort(0, descending=True).values.tolist())

            _scores_pos = pos_scores.sort(0, descending=True).values.tolist()
            for _scores in neg_scores_lists:
                for s1, s2 in itertools.zip_longest(_scores_pos, _scores, fillvalue=100): # fill 100 as a default value
                    if s1 < s2 or s1 == 100:
                        geq_j += 1
                        break
                    elif s1 == s2:
                        continue
                    else:
                        break
            
        for hit in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]:
            if geq_j <= hit:
                hits[hit] += 1
        ranks.append(geq_j)

    hits = {f'hit@{k}':hits[k] for k in sorted(hits)}
    mrr = np.mean(1 / np.array(ranks))

    hits['filtered_mrr'] = mrr
    ipdb.set_trace()

    return hits
def dict_to_cuda(batchdata):
    
    # for batch_key in batchdata:  
    #     for batch in range(len(batchdata[batch_key])) :  
    #         batchdata[batch_key][batch]=batchdata[batch_key][batch].cuda()
    for batch_key in batchdata:
        batchdata[batch_key]=batchdata[batch_key].cuda()
    
    return batchdata
               
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    # ipdb.set_trace()
    # model = AutoModelForSequenceClassification.from_pretrained(
    model = BertForKBCSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    # n_tokens = 10
    # s_wte = SoftEmbedding(model.get_input_embeddings(), 
    #                   n_tokens=n_tokens, 
    #                   initialize_from_vocab=True)
    # ipdb.set_trace()
    # s_wte.load_state_dict(torch.load(os.path.join('./checkpoint/', 'softprompt.pkl')))   
    #         # 用我们设计的类，替换原来的embedding层
    # model.set_input_embeddings(s_wte)

    
    test_dataset = (
            ContextDataset(model_args,data_args,tokenizer,'test')
            if training_args.do_predict
            else None
        )
    # test_dataset = (
    #     GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
    #     if training_args.do_predict
    #     else None
    # )
    
    test_dataloader=DataLoader(test_dataset,
                                batch_size=training_args.per_device_eval_batch_size,
                                collate_fn=myDataset.data_collator,
                                shuffle=False,pin_memory=True)



    if training_args.do_train:
        model = model.to('cuda')
        model.eval()       
                    
        with torch.no_grad():
            logger.info('-----------------------valid step-----------------------')
            # model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_full_neg10_max_inductive/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_full_neg10_max_inductive_test/checkpoint-1000/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_fullxiaorong_nei/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_fulltest0.6/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_testgraph_hop3_full/', 'pytorch_model.bin')))
            model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_full0.1/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_2000_neg10_max_inductive/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_WN18RR_hop3_full_neg10_max_inductive/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_WN18RR_hop3_full_neg10_max_inductive_test/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_WN18RR_hop3_full_neg10_max_inductive_test_2/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_WN18RR_hop3_1000_neg10_max_inductive/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_nell_hop3_full_neg10_max_inductive_test/', 'pytorch_model.bin')))
            
            # model.load_state_dict(torch.load(os.path.join('./output_nell_hop3_full_neg10_max_inductive/', 'pytorch_model.bin')))
            # model.load_state_dict(torch.load(os.path.join('./output_fb237_hop3_full_neg10_max_inductive_new/', 'pytorch_model.bin')))
            
            
            predict = torch.zeros(len(test_dataset), 2, dtype=torch.float)
            # predict = torch.zeros(len(test_dataset), dtype=torch.float)
            
            beginnum=0
            for batch_idx,batchdata in enumerate(tqdm(test_dataloader)):
                for batch_key in batchdata:
                    if batch_key != 'context_text':
                        batchdata[batch_key]=batchdata[batch_key].cuda()
                    else:
                        # ipdb.set_trace()
                        batchdata[batch_key]=dict_to_cuda(batchdata[batch_key])
                eval_output=model(**batchdata)
                eval_len=eval_output.logits.shape[0]
                predict[beginnum:beginnum+eval_len]=eval_output.logits.cpu()
                beginnum+=eval_len
            # ipdb.set_trace()
            
            mrr=compute_ranking_metrics_fn(predict.numpy(),data_args)    
            ipdb.set_trace()


                    
                    
                 

                    # save embedding
                   
                    
              

       
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        
        return 
    # customized early stopping
    # Evaluation
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if output_mode == "classification":
                # write scores of each example into files
                with open(f'{training_args.output_dir}/test_results_prediction_scores.npy', 'wb') as fout:
                    np.save(fout, predictions)
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results
    
if __name__ == "__main__":
    main()