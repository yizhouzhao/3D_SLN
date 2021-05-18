import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

from models.graph import make_mlp

def sample_z(mu, log_var, batch_size = 16, Z_dim = 10):
    # Using reparameterization trick to sample from a gaussian
    eps = torch.randn(batch_size, Z_dim).to(mu.device)
    return mu + torch.exp(log_var / 2) * eps

class TEncoder(nn.Module):
    def __init__(self, 
        embedding_dim,
        Z_dim = 10, # hidden representation size
        Nangle = 24,
    ):
        super().__init__()

        # init info
        # self.vocab = vocab
        num_objs = 32 # len(vocab['object_idx_to_name']) total number of objs in the scene
        # num_preds = len(vocab['pred_idx_to_name'])
        # num_attrs = len(vocab['attrib_idx_to_name'])
        box_embedding_dim = int(embedding_dim * 3 / 4)
        angle_embedding_dim = int(embedding_dim / 4)
        obj_embedding_dim = embedding_dim

        # inicializar redes

        ## embeddings
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.box_embeddings = nn.Linear(6, box_embedding_dim)
        self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)

        ## transform
        self.bert_config = BertConfig()
        self.bert_config.hidden_size = 2 * embedding_dim
        self.bert_encoder = BertEncoder(self.bert_config)
        self.pooler = BertPooler(self.bert_config)

        ## mean var
        self.mean_var = make_mlp([2 * embedding_dim, 64])
        self.hidden_mean = make_mlp([64, Z_dim])
        self.hidden_log_var = make_mlp([64, Z_dim], activation='none')

    def forward(self, objs, boxes, angles, attention_mask):
        '''
        Input:
            objs: stacked object ids [batch x obj_num]
            boxes: stacked boxes [batch x obj_num x 6]
            angles: stacked angles [batch x obj_num]
            attention_mask: attention mask
        '''
        obj_vecs = self.obj_embeddings_ec(objs)
        angle_vecs = self.angle_embeddings(angles)
        boxes_vecs = self.box_embeddings(boxes)

        obj_vecs = torch.cat([obj_vecs, boxes_vecs, angle_vecs], dim=-1)

        bert_outputs = self.bert_encoder(obj_vecs, attention_mask=attention_mask[:, None, None,:])
        pool_outputs = self.pooler(bert_outputs[0])
        mean_var_head_outputs = self.mean_var(pool_outputs)
        mean = self.hidden_mean(mean_var_head_outputs)
        log_var = self.hidden_log_var(mean_var_head_outputs)
        
        return mean, log_var

class GGenerator(nn.Module):
    def __init__(self, input_size,  Z_dim = 10, vocab_size = 32):
        super().__init__()
        self.rnn_cell = nn.GRUCell(input_size, Z_dim)
        self.out = nn.Sequential(
            nn.Linear(Z_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, hx):
        '''
        Input:
            x: the current encoding of the graph
            hx:
        '''
        hx = self.rnn_cell(x, hx)
        output = self.out(hx)

        return output, hx

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()