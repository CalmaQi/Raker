import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import ipdb
import trm

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pytorch_pretrained_bert.modeling import BertEncoder,BertLayerNorm

from typing import List, Mapping, Optional
from transformers.models.bert.modeling_bert import BertConfig,BertEncoder as crossencoder
from transformers import BertPreTrainedModel, BertModel


from transformers.modeling_outputs import (
    SequenceClassifierOutput
)
from transformers import AutoModel,MPNetPreTrainedModel,MPNetModel

class Encoderinput:
    def __init__(self,input_ids,attention_mask,token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

class CrossAttention(nn.Module):
    def __init__(self, args, embedding_dim, num_types):
        super(CrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(embedding_dim, num_types)
        self.device = torch.device('cuda')
        self.embedding_range = 10 / self.embedding_dim
        self.bert_nlayer = 3
        self.bert_nhead = 4
        self.bert_ff_dim = 480
        self.bert_activation = 'gelu'
        self.bert_hidden_dropout =0.2
        self.bert_attn_dropout = 0.2
        self.local_pos_size = 12
        self.bert_layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)
        
        bert_config = BertConfig(0, hidden_size=self.embedding_dim,
                                 num_hidden_layers=self.bert_nlayer // 2,
                                 num_attention_heads=self.bert_nhead,
                                 intermediate_size=self.bert_ff_dim,
                                 hidden_act=self.bert_activation,
                                 hidden_dropout_prob=self.bert_hidden_dropout,
                                 attention_probs_dropout_prob=self.bert_attn_dropout,
                                 max_position_embeddings=0,
                                 type_vocab_size=0,
                                 initializer_range=self.embedding_range,
                                 add_cross_attention=True,
                                 is_decoder=True)
        self.bert_encoder = crossencoder(bert_config)
        # self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Linear(embedding_dim,num_types)      

        
    def forward(self,  hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        head_mask=None):
        
        output=self.bert_encoder(hidden_states=hidden_states,attention_mask=attention_mask,
                                 encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=encoder_attention_mask,output_attentions=True)
       
        # predict=self.classifier()
        return output    
class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """这个类用来给模型附加一个用于学习的embedding
        Args:
            wte (nn.Embedding): 这个参数，是预训练模型的embedding，载入进来用来提取一些参数。
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens, 
                                                                                  random_range, 
                                                                                  initialize_from_vocab))

    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """初始化学习向量
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        # 有两种初始化方式，一种是从预训练模型copy一部分token，进行训练
        # 另一种是随机生成一部分训练
        # 结果上来说区别不大
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        # 把我们新加入的固定长度的，用于代表任务的prompt embedding，和实际的embedding合并
        return torch.cat([learned_embedding, input_embedding], 1)
 
class BertForKBCSequenceClassification(BertPreTrainedModel):
# class BertForKBCSequenceClassification(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # self.mpnet = MPNetModel(config)
        self.bert = BertModel(config)
        # config.is_decoder=True
        # config.add_cross_attention=True
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)      
        # self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)     
        self.ntoken=10
        self.tntoken=10
         
        self.seq_indices_relation = torch.LongTensor(list(range(self.ntoken)))
        self.tseq_indices_relation = torch.LongTensor(list(range(self.tntoken)))
        # embedding
        # self.embedding_relation = torch.nn.Embedding(self.ntoken * 10, config.hidden_size)
        
        self.embedding_relation = torch.nn.Embedding(self.ntoken * 238, config.hidden_size)
        self.type_relation = torch.nn.Embedding(self.tntoken *4, config.hidden_size)
        # LSTM

        self.init_weights()
        # self.crossencoder=CrossAttention(config,config.hidden_size,config.num_labels)
        # self.contextlayer=crossencoder(config,config.hidden_size,config.num_labels,1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        context_text=None,
        rs_tensor=None,
        rt_tensor=None,
        mode='train1',
        
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ipdb.set_trace()
        

        # ipdb.set_trace()

        
        rt_tensor=self.get_rtype(rs_tensor)
        #  bz x template
        seq_indices_relation_spec = self.seq_indices_relation.cuda().unsqueeze(0) + rs_tensor.unsqueeze(-1) * self.ntoken
        tseq_indices_relation= self.tseq_indices_relation.cuda().unsqueeze(0) + rt_tensor.unsqueeze(-1) * self.tntoken
        # bz x template x dim
        prompt_emb = self.embedding_relation(seq_indices_relation_spec)
        tprompt_emb=self.type_relation(tseq_indices_relation)
        raw_emb=self.bert.get_input_embeddings()(input_ids)
        if mode is 'train':
            noise_alpha=5
            dims = torch.tensor(raw_emb.size(1) * raw_emb.size(2))
            mag_norm = noise_alpha/torch.sqrt(dims)
            raw_emb=raw_emb + torch.zeros_like(raw_emb).uniform_(-mag_norm, mag_norm)
        if self.ntoken:
            raw_emb[:,1:1+self.ntoken]=prompt_emb[:,]
            raw_emb[:,1+self.ntoken:1+self.ntoken+self.tntoken]=tprompt_emb[:,]
        
        input_embeds=raw_emb
        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # outputs = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
       
       
        
        # context_output=torch.cat([context_output.squeeze(1),pooled_output],dim=1)
        logits = self.classifier(pooled_output)
        
       
        loss = None
        if labels is not None:
            # transform labels to probability label B x 2
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # pos_loss=(1-cossim)*labels
                # neg_loss=(1-labels)*torch.clamp(cossim,min=0)             
                # loss=torch.sum(pos_loss+neg_loss)/labels.shape[0]                
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # loss = loss_fct(logits.view(-1), labels.view(-1))
                
                
                

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,

        )
        
    def get_rtype(self, rs_tensor):
        rt_tensor = torch.ones(rs_tensor.shape, dtype=torch.long).cuda()
        rs_rt = {}
        with open('./r_type.txt') as f:
            for i in f:
                rs, rt = i.strip().split()
                rs_rt[int(rs)] = [int(rt)]
        for i in range(len(rt_tensor)):
            rt_tensor[i] = torch.tensor(rs_rt[int(rs_tensor[i])], dtype=torch.long).cuda()

        return rt_tensor  
    def context_encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output
    
def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector

class ContextLayer(nn.Module):
    def __init__(self, args, embedding_dim, num_types, temperature):
        super(ContextLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(embedding_dim, num_types)
        self.temperature = temperature
        self.device = torch.device('cuda')

        self.trm_nlayer = 3 # args['trm_nlayer']
        self.trm_nhead = 4 # args['trm_nhead']
        self.trm_hidden_dropout = 0.2 #args['trm_hidden_dropout']
        self.trm_attn_dropout =0.2 # args['trm_attn_dropout']
        self.trm_ff_dim = 768 # args['trm_ff_dim']
        self.global_pos_size = 150 # args['global_pos_size']
        self.embedding_range = 10 / self.embedding_dim

        self.global_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.global_cls, std=self.embedding_range)
        self.global_sep = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.global_sep, std=self.embedding_range)
        self.global_pad = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.global_pad, std=self.embedding_range)
        
        self.pos_embeds = nn.Embedding(self.global_pos_size, self.embedding_dim)
        torch.nn.init.normal_(self.pos_embeds.weight, std=self.embedding_range)
        
        self.layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)

        self.transformer_encoder = trm.Encoder(
            lambda: trm.EncoderLayer(
                self.embedding_dim,
                trm.MultiHeadedAttentionWithRelations(
                    self.trm_nhead,
                    self.embedding_dim,
                    self.trm_attn_dropout),
                trm.PositionwiseFeedForward(
                    self.embedding_dim,
                    self.trm_ff_dim,
                    self.trm_hidden_dropout),
                num_relation_kinds=0,
                dropout=self.trm_hidden_dropout),
            self.trm_nlayer,
            self.embedding_range,
            tie_layers=False)

    def convert_mask_trm(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def forward(self, local_embedding):
        
        batch_size= len(local_embedding)
        for batch_iter in range(batch_size):
            neighbor_size, emb_size=local_embedding[batch_iter].shape
            if neighbor_size<7:
                local_embedding[batch_iter]=torch.cat([local_embedding[batch_iter],self.global_pad.expand(7-neighbor_size, emb_size)])
        
        local_embedding=torch.cat([i.unsqueeze(0) for i  in local_embedding])
        
        #对localembedding预处理 做padding
        batch_size, neighbor_size, emb_size = local_embedding.size()
        
        attention_mask = torch.ones(batch_size, neighbor_size + 2).bool().to(self.device)
        second_local = torch.cat([self.global_cls.expand(batch_size, 1, emb_size), local_embedding], dim=1)
        #加sep
        # ipdb.set_trace()
        second_local = torch.cat([second_local[:,:2],self.global_sep.expand(batch_size, 1, emb_size),second_local[:,2:]], dim=1)
        
        pos = self.pos_embeds(torch.arange(0, 4).to(self.device))
        second_local[:, 0] = second_local[:, 0] + pos[0].unsqueeze(0)
        second_local[:, 1] = second_local[:, 1] + pos[1].unsqueeze(0)
        second_local[:, 2] = second_local[:, 2] + pos[2].unsqueeze(0)
        second_local[:, 3:] = second_local[:, 3:] + pos[3].view(1, 1, -1)
        
        second_local = self.layer_norm(second_local)
        second_local = self.transformer_encoder(second_local, None, self.convert_mask_trm(attention_mask))
        second_local = second_local[-1][:, :2][:, 0].unsqueeze(1)
        # return torch.relu(second_local)
        # ipdb.set_trace()
        # predict = self.fc((second_local))
        
        predict = self.fc(torch.relu(second_local))
        # predict2 = self.fc(torch.relu(second_local))
        # predict3 = self.fc(torch.relu(global_embedding))

        # predict = torch.cat([predict1, predict2, predict3], dim=1)
        # weight = torch.softmax(self.temperature * predict, dim=1)
        # predict = (predict * weight.detach()).sum(1).sigmoid()

        return predict



class ContextLayer2(nn.Module):
    def __init__(self, args, embedding_dim, num_types, temperature):
        super(ContextLayer2, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(embedding_dim, num_types)
        self.temperature = temperature
        self.device = torch.device('cuda')
        self.embedding_range = 10 / self.embedding_dim

        self.bert_nlayer = 3
        self.bert_nhead = 4
        self.bert_ff_dim = 480
        self.bert_activation = 'gelu'
        self.bert_hidden_dropout =0.2
        self.bert_attn_dropout = 0.2
        self.local_pos_size = 12
        self.bert_layer_norm = BertLayerNorm(self.embedding_dim, eps=1e-12)
        self.global_pad = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.global_pad, std=self.embedding_range)
        self.global_sep = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.global_sep, std=self.embedding_range)
        self.local_cls = nn.Parameter(torch.Tensor(1, self.embedding_dim))
        torch.nn.init.normal_(self.local_cls, std=self.embedding_range)
        self.local_pos_embeds = nn.Embedding(self.local_pos_size, self.embedding_dim)
        torch.nn.init.normal_(self.local_pos_embeds.weight, std=self.embedding_range)
        bert_config = BertConfig(0, hidden_size=self.embedding_dim,
                                 num_hidden_layers=self.bert_nlayer // 2,
                                 num_attention_heads=self.bert_nhead,
                                 intermediate_size=self.bert_ff_dim,
                                 hidden_act=self.bert_activation,
                                 hidden_dropout_prob=self.bert_hidden_dropout,
                                 attention_probs_dropout_prob=self.bert_attn_dropout,
                                 max_position_embeddings=0,
                                 type_vocab_size=0,
                                 initializer_range=self.embedding_range)
        self.bert_encoder = BertEncoder(bert_config)

    def convert_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

    def forward(self, local_embedding):

        batch_size= len(local_embedding)
        for batch_iter in range(batch_size):
            neighbor_size, emb_size=local_embedding[batch_iter].shape
            if neighbor_size<7:
                local_embedding[batch_iter]=torch.cat([local_embedding[batch_iter],self.global_pad.expand(7-neighbor_size, emb_size)])
                
        # et_merge = torch.cat([et_types, et_relations], dim=1).view(-1, 2, self.embedding_dim)
        et_merge=torch.cat([i.unsqueeze(0) for i  in local_embedding])

        et_pos = self.local_pos_embeds(torch.arange(0, 9, device=self.device)).unsqueeze(0).repeat(et_merge.shape[0], 1, 1)
        et_merge = torch.cat([self.local_cls.expand(et_merge.size(0), 1, self.embedding_dim), et_merge], dim=1)
        
        et_merge = torch.cat([et_merge[:,:2],self.global_sep.expand(batch_size, 1, emb_size),et_merge[:,2:]], dim=1)
        
        et_merge=et_merge+ et_pos
        ipdb.set_trace()
        et_merge = self.bert_layer_norm(et_merge)
        #batch_size maxlen dim
        
        et_merge = self.bert_encoder(et_merge, self.convert_mask(et_merge.new_ones(et_merge.size(0), et_merge.size(1), dtype=torch.long)),
                                     output_all_encoded_layers=False)[-1][:, 0].view(batch_size, -1, self.embedding_dim)
        
        
        predict = self.fc((et_merge))
        # predict2 = self.fc(torch.relu(second_local))
        # predict3 = self.fc(torch.relu(global_embedding))

        # predict = torch.cat([predict1, predict2, predict3], dim=1)
        # weight = torch.softmax(self.temperature * predict, dim=1)
        # predict = (predict * weight.detach()).sum(1).sigmoid()

        return predict



class SentenceModel(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = MPNetModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)      

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        context_text=None,
        rs_tensor=None,
        
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ipdb.set_trace()
        maxlen=30
        output1=self.bert(
                input_ids=context_text['input_ids'][:,:maxlen],
                attention_mask=context_text['attention_mask'][:,:maxlen],
                return_dict=True)
        output2=self.bert(
                input_ids=context_text['input_ids'][:,maxlen:],
                attention_mask=context_text['attention_mask'][:,maxlen:],
                return_dict=True)
        # ipdb.set_trace()
         
        def mean_pooling(model_output,attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        output1_emb=mean_pooling(output1,context_text['attention_mask'][:,:maxlen])    
        output1_emb = F.normalize(output1_emb, p=2, dim=1)
        output2_emb=mean_pooling(output2,context_text['attention_mask'][:,maxlen:])  
        output2_emb = F.normalize(output2_emb, p=2, dim=1)
        
        cossim=torch.cosine_similarity(output1_emb, output2_emb, dim=1)  
        
        
        logits=cossim
        loss = None
        if labels is not None:
            # transform labels to probability label B x 2
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                pos_loss=(1-cossim)*labels
                neg_loss=(1-labels)*torch.clamp(cossim,min=0)
                
                
                loss=torch.sum(pos_loss+neg_loss)/labels.shape[0]
                
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,

        )