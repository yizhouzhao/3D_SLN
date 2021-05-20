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
from utils import calculate_model_losses

args = Options().parse()
if (args.output_dir is not None) and (not os.path.isdir(args.output_dir)):
    os.mkdir(args.output_dir)
if (args.test_dir is not None) and (not os.path.isdir(args.test_dir)):
    os.mkdir(args.test_dir)

# no KL divergence loss
args.use_AE = True

# tensorboard
writer = SummaryWriter()

# load data
vocab, train_loader, val_loader = build_loaders(args)

dt = train_loader.dataset

# load decoder
ovaed = OriVAEDecoder(dt.vocab, embedding_dim=64)
ovaed = ovaed.cuda()

optimizer = torch.optim.Adam(ovaed.parameters(), lr=args.learning_rate)

t = 0 # total steps

for epoch in range(5):
    print("Trainaing epoch {}".format(epoch))
    # training
    ovaed.train()
    for batch in tqdm(train_loader):
        t += 1
        
        ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)
        z = torch.randn(objs.size(0), 64).to(objs.device)
        boxes_pred, angles_pred = ovaed.decoder(z, objs, triples, attributes)
        
        total_loss, losses = calculate_model_losses(args, ovaed, boxes_pred, boxes, angles, angles_pred)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if t % args.print_every == 0:
            print("On batch {} in epoch {}".format(t, epoch))
            for name, val in losses.items():
                print(' [%s]: %.4f' % (name, val))
                writer.add_scalar('Loss/'+ name, val, t)

    # validation
    ovaed.eval()
    valid_loss_list = {"bbox_pred":[], "angle_pred": []}
    for batch in tqdm(val_loader):        
        ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)
        z = torch.randn(objs.size(0), 64).to(objs.device)
        boxes_pred, angles_pred = ovaed.decoder(z, objs, triples, attributes)
        
        total_loss, losses = calculate_model_losses(args, ovaed, boxes_pred, boxes, angles, angles_pred)
        
        #if t % args.print_every == 0:
        #    print("On batch {} out of {}".format(t, args.num_iterations))
        for name, val in losses.items():
            valid_loss_list[name].append(val.item())

        for name, val_list in valid_loss_list.items():
            writer.add_scalar('Loss/Validation_'+ name, np.mean(val_list), epoch)
    
        
