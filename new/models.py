from logging import makeLogRecord
from new.networks import *

class FromEncoderToGraphGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        # params
        self.embedding_dim = 168
        self.Z_dim = 10
        self.vocab_size = 32 # start token: 32

        # networks
        self.encoder = TEncoder(self.embedding_dim, self.Z_dim)
        
        self.graph_generator =  GGenerator(input_size = 64, hidden_dim = 64, 
                                        vocab_size = 32, Z_dim = self.Z_dim) 

        # self.graph_encoder = GraphEncoder()
        self.obj_embedding_gg = nn.Embedding(num_embeddings = self.vocab_size + 1, embedding_dim=64)
        self.obj_pred_gg = make_mlp([64, self.vocab_size + 1])

        # loss
        self.language_loss = nn.CrossEntropyLoss()
        
    def forward(self, objs, boxes, angles, attention_mask):
        # load data
        # = batch[0]
        # graph_batch = batch[1]
        # encoder
        mean, log_var = self.encoder(objs, boxes, angles, attention_mask)

        # resampling
        z_sampled = sample_z(mean, log_var)

        # graph generator
        hx = self.graph_generator.init_hidden_states(z_sampled)

        # training language model
        pad_start = torch.ones(objs.size(0), 1).long().to(objs.device) * 32
        pad_inputs = torch.cat([pad_start,objs],dim=1)

        all_logits = []
        for i in range(objs.size(1)): # length
            obj_embeded_gg = self.obj_embedding_gg(pad_inputs[:,i])
            gg_output, hx = self.graph_generator(obj_embeded_gg, hx)
            all_logits.append(gg_output)

        all_logits = torch.stack(all_logits, dim = 1)
        # print("all_logits", all_logits.shape)
        logits = all_logits.view(-1, self.vocab_size + 1)
        # print("logits", logits.shape)
        target = pad_inputs[:,1:].flatten()
        # print("target", target.shape)
        
        loss = self.language_loss(logits, target)

        return all_logits, loss



