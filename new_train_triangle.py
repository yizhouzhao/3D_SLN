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
from new.utils import resolve_relative_positions, obtain_sampled_relations
from utils import calculate_model_losses
from data.suncg_dataset import g_add_in_room_relation, g_use_heuristic_relation_matrix, \
    g_prepend_room, g_add_random_parent_link, g_shuffle_subject_object

# from torch_scatter import scatter_mean

# decoder option
g_decoder_option = "original" #"rgcn"
g_relative_location = "False" # FIX FLASE CURRENTLY
g_parent_link_index = 16

args = Options().parse()
if (args.output_dir is not None) and (not os.path.isdir(args.output_dir)):
    os.mkdir(args.output_dir)
if (args.test_dir is not None) and (not os.path.isdir(args.test_dir)):
    os.mkdir(args.test_dir)

# change embedding_dim for bert
args.embedding_dim = 84 
# has KL divergence loss
args.use_AE = False

# tensorboard
writer = SummaryWriter()
writer.add_text("args", str(args))

writer.add_hparams({
    "experiment type": "Trianglur Training Decay Training: Transformer Encoder + Original Decoder + Graph Generator",
    "decoder type": g_decoder_option,
    "use relative location": g_relative_location,
    "Add 'in_room' relation": g_add_in_room_relation,
    "Use heuristic relation matrix": g_use_heuristic_relation_matrix,
    "prepend/append room info": g_prepend_room,
    "add random parent link": g_add_random_parent_link,
    "shuffle object/subject when loading data": g_shuffle_subject_object,
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

# load encoder
model_encoder = TransformerEncoder(**kwargs)
model_encoder = model_encoder.cuda()
optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=args.learning_rate)

# load decoder
if g_decoder_option == "original":
    model_decoder = OriVAEDecoder(**kwargs)
elif g_decoder_option == "rgcn":
    model_decoder = RGCNConv(**kwargs)
else:
    raise("MODEL MISSING {}".format(g_decoder_option))
model_decoder = model_decoder.cuda()
optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=args.learning_rate) 

# load graph generator
model_generator = GraphGenerator(**kwargs)
model_generator = model_generator.cuda()
optimizer_generator = torch.optim.Adam(model_generator.parameters(), lr=args.learning_rate) 

t = 0 # total steps
total_epochs = 10

for epoch in range(total_epochs):
    print("Training epoch {}".format(epoch))
    # training
    model_encoder.train()
    model_decoder.train()
    for batch in tqdm(train_loader):
        t += 1
        # break
        try:
            ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)

            # attention mask
            obj_counts = [torch.sum(obj_to_img == i).item() for i in range(args.batch_size)]
            block_list = [torch.ones((obj_counts[i],obj_counts[i])) for i in range(args.batch_size)]
            attention_mask = torch.block_diag(*block_list).to(objs.device) # [BxB]

            # encoder
            hidden_states = model_encoder.encoder(objs, boxes, angles, attributes, attention_mask)
            mu, logvar = model_encoder.get_hidden_representation(hidden_states)

            # re-parameterization
            if args.use_AE:
                z = mu
            else:
                # reparameterization
                std = torch.exp(0.5*logvar)
                # standard sampling
                eps = torch.randn_like(std)
                z = eps.mul(std).add_(mu)

            # KL weight
            if args.KL_linear_decay:
                    KL_weight = 10 ** (t // 1e5 - 6)
            else:
                KL_weight = args.KL_loss_weight
            
            if np.random.randn() > (epoch + 1) / (total_epochs + 1):
                train_type = "vae"
            else:
                train_type = "rl"

            if train_type == "vae":
                # decoder
                boxes_pred, angles_pred = model_decoder.decoder(z, objs, triples, attributes)
                total_loss, losses = calculate_model_losses(args, None, boxes, boxes_pred, angles, angles_pred, mu=mu, logvar=logvar, KL_weight=KL_weight)
                losses['total_loss'] = total_loss.item()
                
                if not math.isfinite(losses['total_loss']):
                    print('WARNING: Got loss = NaN, not backpropping')
                    #continue
                
                if g_relative_location:
                    boxes_pred = resolve_relative_positions(boxes_pred, triples, g_parent_link_index)

                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                total_loss.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()

                if t % args.print_every == 0:
                    print("On batch {} in epoch {}".format(t, epoch))
                    for name, val in losses.items():
                        print(' [%s]: %.4f' % (name, val))
                        writer.add_scalar('Loss/'+ name, val, t)
            
            if train_type == "rl":
                # calculate edges
                score_matrix = model_generator.get_score_matrix(hidden_states, attention_mask)
                all_samples, all_log_probs, all_entropy = model_generator.sample(score_matrix, obj_to_img)

                # query new relation
                new_triples = obtain_sampled_relations(objs, all_samples, boxes.cpu().data, dt.vocab)

                # decoder
                boxes_pred, angles_pred = model_decoder.decoder(z, objs, new_triples, attributes)

                # loss
                loss_bbox = F.l1_loss(boxes_pred, boxes, reduction = "none")
                loss_bbox = torch.mean(loss_bbox, dim = 1)
                #loss_bbox_per_batch = scatter_mean(loss_bbox, obj_to_img, dim = 0)

                loss_angle = F.nll_loss(angles_pred, angles, reduction = "none")
                #loss_angle_per_batch = scatter_mean(loss_angle, obj_to_img, dim = 0)

                # calculate policy gradient
                J = - torch.mean(all_log_probs * (loss_bbox.detach() + \
                        args.rl_angle_loss_weight * loss_angle.detach()) + \
                        args.entropy_alpha * all_entropy)

                optimizer_generator.zero_grad()
                optimizer_decoder.zero_grad()
                J.backward()
                optimizer_generator.step()
                optimizer_decoder.step()

                if t % args.print_every == 0:
                    print("On batch {} in epoch {} RL Part".format(t, epoch))
                    print(' [%s]: %.4f' % ("RL mean reward", J.item()))
                    writer.add_scalar('Loss/'+ "RL_mean_reward", J.item(), t)

                    # then calculate reconstruction loss
                    total_loss, losses = calculate_model_losses(args, None, boxes, boxes_pred, angles, angles_pred, mu=mu, logvar=logvar, KL_weight=KL_weight)
                    losses['total_loss'] = total_loss.item()
                    
                    for name, val in losses.items():
                        print(' [%s]: %.4f' % (name, val))
                        writer.add_scalar('Loss/RL_'+ name, val, t)
        except:
            continue

    # validation
    model_encoder.eval()
    model_decoder.eval()
    print("Validation epoch {}".format(epoch))
    valid_loss_list = {"bbox_pred":[], "angle_pred": [],"total_loss":[], 'KLD_Gauss':[],
                        "rl_reward":[], "rl_angle_pred":[], "rl_box_pred":[]}

    for batch in tqdm(val_loader):       
        try:
            ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)
            
            # attention mask
            obj_counts = [torch.sum(obj_to_img == i).item() for i in range(args.batch_size)]
            block_list = [torch.ones((obj_counts[i],obj_counts[i])) for i in range(args.batch_size)]
            attention_mask = torch.block_diag(*block_list).to(objs.device) # [BxB]

            # encoder
            hidden_states = model_encoder.encoder(objs, boxes, angles, attributes, attention_mask)
            mu, logvar = model_encoder.get_hidden_representation(hidden_states)

            # re-parameterization
            if args.use_AE:
                z = mu
            else:
                # reparameterization
                std = torch.exp(0.5*logvar)
                # standard sampling
                eps = torch.randn_like(std)
                z = eps.mul(std).add_(mu)

            # decoder
            boxes_pred, angles_pred = model_decoder.decoder(z, objs, triples, attributes)
            
            if args.KL_linear_decay:
                KL_weight = 10 ** (t // 1e5 - 6)
            else:
                KL_weight = args.KL_loss_weight
            total_loss, losses = calculate_model_losses(args, None, boxes, boxes_pred, angles, angles_pred, mu=mu, logvar=logvar, KL_weight=KL_weight)
            losses['total_loss'] = total_loss.item()
            
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                #continue
            
            for name, val in losses.items():
                valid_loss_list[name].append(val)
            
            # RL part
            # calculate edges
            score_matrix = model_generator.get_score_matrix(hidden_states, attention_mask)
            all_samples, all_log_probs, all_entropy = model_generator.sample(score_matrix, obj_to_img)

            # query new relation
            new_triples = obtain_sampled_relations(objs, all_samples, boxes.cpu().data, dt.vocab)

            # decoder
            rl_boxes_pred, rl_angles_pred = model_decoder.decoder(z, objs, new_triples, attributes)

            # loss
            loss_bbox = F.l1_loss(boxes_pred, boxes, reduction = "none")
            loss_bbox = torch.mean(loss_bbox, dim = 1)
            #loss_bbox_per_batch = scatter_mean(loss_bbox, obj_to_img, dim = 0)

            loss_angle = F.nll_loss(angles_pred, angles, reduction = "none")
            #loss_angle_per_batch = scatter_mean(loss_angle, obj_to_img, dim = 0)

            # calculate policy gradient
            J = - torch.mean(all_log_probs * (loss_bbox.detach() + \
                    args.rl_angle_loss_weight * loss_angle.detach()) + \
                    args.entropy_alpha * all_entropy)
            valid_loss_list["rl_reward"].append(J.item())

            # then calculate reconstruction loss
            rl_total_loss, rl_losses = calculate_model_losses(args, None, boxes, rl_boxes_pred, angles, rl_angles_pred, mu=mu, logvar=logvar, KL_weight=KL_weight)
            valid_loss_list["rl_box_pred"].append(rl_losses['bbox_pred'])
            valid_loss_list["rl_angle_pred"].append(rl_losses['angle_pred'])      
        except:
            continue  
        
    for name, val_list in valid_loss_list.items():
        writer.add_scalar('Loss/Validation_'+ name, np.mean(val_list), epoch)
        print("Validation loss", name, np.mean(val_list))
    
torch.save(model_encoder.state_dict(), "records/encoder_May24.pth")
torch.save(model_decoder.state_dict(), "records/decoder_May24.pth")
torch.save(model_generator.state_dict(), "records/model_generator_May24.pth")