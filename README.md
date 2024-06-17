# Raker: A Relation-aware Knowledge Reasoning Model for Inductive Relation Prediction
Code and data for  paper Raker: A Relation-aware Knowledge Reasoning Model for Inductive Relation Prediction.

## Requirements:
- huggingface transformer
- pytorch
- networkx
- tqdm
- numpy
- sklearn
- ipdb

## Download the Dataset Split
Here we provide the data split used in paper in folder "data". The $DATASET$PART and $DATASET$PART_ind contain corresponding transductive and inductive subgraphs. 
Each train/valid/test file contains a list of knowledge graph triples. "ranking_head.txt" and "ranking_tail.txt" are presampled candidates 
for the predicting the missing tail triple and missing head triple in knowledge graph completion. Each triple contains 50 candidates for tail and 50 for head in this file.
$DATASET denotes the dataset name, and $PART denotes the size of the dataset, whether it is a fewshot version or full. If $PART is not specified, it is full by default.


## Preprocessing Data
folder "bertrl_data" provides an example of preprocessed data to be input of the model. 
<!-- They are actual tsv data examples. Here we show the example preprocessing scripts. $DATASET denotes the name of the dataset in folder "data", e.g. fb237. -->
part paramerter can be specified as full, 1000, 2000, referring to folder "data".

```
python load_data.py -d $DATASET -st test --part full --hop 3 --ind_suffix "_ind" --suffix "_neg10_max_inductive"
```

## Raker
1. Training model
We provide example bash scripts in train.sh

