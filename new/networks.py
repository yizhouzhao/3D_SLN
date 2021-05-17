import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from models.graph import make_mlp

class TEncoder(nn.Module):
    def __init__(self, 
        vocab, 
        embedding_dim,
        encoder_hidden_dim = 10, # hidden representation size
        Nangle = 24,
    ):
        super().__init__()

        # init info
        self.vocab = vocab
        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        num_attrs = len(vocab['attrib_idx_to_name'])
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

        ## mean var
        self.mean_var = make_mlp([2 * embedding_dim, 64])
        self.hidden_mean = make_mlp([64, encoder_hidden_dim])
        self.hidden_log_var = make_mlp([64, encoder_hidden_dim], activation='none')


    def forward(self, objs, boxes, angles, batch_lengths):
        '''
        Input:
            objs: stacked object ids [# of all objects in the batch]
            boxes: stacked boxes [# of all objects in the batch x 6]
            angles: stacked angles [# of all objects in the batch]
            batch_lengths: a list of object counts in the batch [batch_size]: e.g. [6, 8, 8,...]
        '''
        obj_vecs = self.obj_embeddings_ec(objs)
        angle_vecs = self.angle_embeddings(angles)
        boxes_vecs = self.box_embeddings(boxes)

        obj_vecs = torch.cat([obj_vecs, boxes_vecs, angle_vecs], dim=1)

        transformer_outputs = self.bert_encoder(obj_vecs)


        

