from options.options import Options
import os
import torch
from build_dataset_model import build_loaders, build_model
from utils import get_model_attr, calculate_model_losses, tensor_aug
from collections import defaultdict
import math

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np

from new.CustomVAE import *
from new.utils import resolve_relative_positions
from utils import calculate_model_losses
from data.suncg_dataset import g_add_in_room_relation, g_use_heuristic_relation_matrix, g_prepend_room, g_add_random_parent_link

# decoder option
g_decoder_option = "rgcn"
g_relative_location = "true"
g_parent_link_index = 16

args = Options().parse()
if (args.output_dir is not None) and (not os.path.isdir(args.output_dir)):
    os.mkdir(args.output_dir)
if (args.test_dir is not None) and (not os.path.isdir(args.test_dir)):
    os.mkdir(args.test_dir)

# no KL divergence loss
args.use_AE = True

# tensorboard
writer = SummaryWriter()

writer.add_hparams({
    "experiment type": "Decoder update: add random parent link for relative position calculation",
    "decoder type": g_decoder_option,
    "use relative location": g_relative_location,
    "Add 'in_room' relation": g_add_in_room_relation,
    "Use heuristic relation matrix": g_use_heuristic_relation_matrix,
    "prepend/append room info": g_prepend_room,
    "add random parent link": g_add_random_parent_link,
}, {"NA": 0})

# load data
vocab, train_loader, val_loader = build_loaders(args)

dt = train_loader.dataset

# model args
kwargs = {
        'vocab': dt.vocab,
        'batch_size': args.batch_size,
        'train_3d': args.train_3d,
        'decoder_cat': args.decoder_cat,
        'embedding_dim': args.embedding_dim,
        'gconv_mode': args.gconv_mode,
        'gconv_num_layers': args.gconv_num_layers,
        'mlp_normalization': args.mlp_normalization,
        'vec_noise_dim': args.vec_noise_dim,
        'layout_noise_dim': args.layout_noise_dim,
        'use_AE': args.use_AE
    }

# load decoder
if g_decoder_option == "rgcn":
    model = OriVAEDecoder(dt.vocab, embedding_dim=64)
else:
    model = RGCNConv(**kwargs)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

t = 0 # total steps

for epoch in range(5):
    print("Training epoch {}".format(epoch))
    # training
    model.train()
    for batch in tqdm(train_loader):
        t += 1
        ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)
        z = torch.randn(objs.size(0), 64).to(objs.device)
        boxes_pred, angles_pred = model.decoder(z, objs, triples, attributes)
        
        if g_relative_location:
            boxes_pred = resolve_relative_positions(boxes_pred, triples, g_parent_link_index)

        total_loss, losses = calculate_model_losses(args, model, boxes_pred, boxes, angles, angles_pred)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if t % args.print_every == 0:
            print("On batch {} in epoch {}".format(t, epoch))
            for name, val in losses.items():
                print(' [%s]: %.4f' % (name, val))
                writer.add_scalar('Loss/'+ name, val, t)

    # validation
    model.eval()
    print("Validation epoch {}".format(epoch))
    valid_loss_list = {"bbox_pred":[], "angle_pred": []}
    for batch in tqdm(val_loader):        
        ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)
        z = torch.randn(objs.size(0), 64).to(objs.device)
        boxes_pred, angles_pred = model.decoder(z, objs, triples, attributes)
        
        if g_relative_location:
            boxes_pred = resolve_relative_positions(boxes_pred, triples, g_parent_link_index)

        total_loss, losses = calculate_model_losses(args, model, boxes_pred, boxes, angles, angles_pred)
        
        #if t % args.print_every == 0:
        #    print("On batch {} out of {}".format(t, args.num_iterations))
        for name, val in losses.items():
            valid_loss_list[name].append(val)

    for name, val_list in valid_loss_list.items():
        writer.add_scalar('Loss/Validation_'+ name, np.mean(val_list), epoch)
        print("Validation loss", name, np.mean(val_list))
    
        
