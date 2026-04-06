import torch
import pickle
from torch_geometric.data import Data
from tqdm import tqdm
from app import normalization
from DSTPP.Models import build_graph
import os
from app import get_args

opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

# Import Data
f = open('dataset/{}/data_{}.pkl'.format(opt.dataset, opt.type), 'rb')
raw_data = pickle.load(f) 
raw_data = [[list(i) for i in u] for u in raw_data]
raw_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in raw_data] # [time, delta_time, lat, long]

Max, Min = [], []
for m in range(4):
    if m > 0:
        Max.append(max([i[m] for u in raw_data for i in u]))
        Min.append(min([i[m] for u in raw_data for i in u]))
    else:
        Max.append(1)
        Min.append(0)
assert Min[1] >= 0
raw_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in raw_data]

# Convert to tensor
event_time = [torch.tensor([e[1] for e in seq], dtype=torch.float32) for seq in raw_data]
event_loc = [torch.tensor([e[2:] for e in seq], dtype=torch.float32) for seq in raw_data]

# Build Graph
graph_list = build_graph(event_time, event_loc, connect_ratio = 0.1)

save_path = "GraphData/{}_graphs_{}.pt".format(opt.dataset, opt.type)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(graph_list, save_path)

# run code
# python precompute_graph_gnn.py --dataset $dataset --type $type
# e.g. python precompute_graph_gnn.py --dataset US_Earthquake --type train
