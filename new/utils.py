# utils
import torch 
from torch.nn.utils.rnn import pad_sequence

def new_collate_fn(batch):
    all_mask = [] # attention mask
    all_objs = []
    all_boxes = []
    all_angles = []
    for i, (room_id, objs, boxes, triples, angles, attributes) in enumerate(batch):
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_angles.append(angles)
        all_mask.append(torch.ones(objs.size(0)).long())
    
    pad_all_mask = pad_sequence(all_mask, padding_value=0, batch_first=True) # [B x O]
    pad_all_objs = pad_sequence(all_objs, padding_value=0, batch_first=True) # [B x O]
    pad_all_boxes = pad_sequence(all_boxes, padding_value=0, batch_first=True) # [B x O x 6]
    pad_all_angles = pad_sequence(all_angles, padding_value=0, batch_first=True) # [B x O]

    return pad_all_objs, pad_all_boxes, pad_all_angles, pad_all_mask