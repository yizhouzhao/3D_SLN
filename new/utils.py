# utils
import torch 
from torch.nn.utils.rnn import pad_sequence

from torch_geometric.data import Data, Batch


def new_collate_fn(batch):
    all_mask = [] # attention mask
    all_objs = []
    all_boxes = []
    all_angles = []

    all_graph_data = []
    for i, (room_id, objs, boxes, triples, angles, attributes) in enumerate(batch):
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_angles.append(angles)
        all_mask.append(torch.ones(objs.size(0)).long())

        graph_x = objs.unsqueeze(1)
        graph_edge_index = triples[:,[0,2]].transpose(0, 1)
        graph_edge_attr = triples[:, [1]]

        graph_data = Data(graph_x, edge_index=graph_edge_index, edge_attr=graph_edge_attr)
        all_graph_data.append(graph_data)
    
    pad_all_mask = pad_sequence(all_mask, padding_value=0, batch_first=True) # [B x O]
    pad_all_objs = pad_sequence(all_objs, padding_value=0, batch_first=True) # [B x O]
    pad_all_boxes = pad_sequence(all_boxes, padding_value=0, batch_first=True) # [B x O x 6]
    pad_all_angles = pad_sequence(all_angles, padding_value=0, batch_first=True) # [B x O]

    graph_batch = Batch.from_data_list(all_graph_data)

    return (pad_all_objs, pad_all_boxes, pad_all_angles, pad_all_mask), graph_batch