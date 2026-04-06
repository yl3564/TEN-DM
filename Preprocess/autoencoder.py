import os
from tqdm import tqdm
import torch
from torch_geometric.nn import GAE, GATConv
from torch_geometric.transforms import RandomLinkSplit
from app import get_args

opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

transform = RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    split_labels=True,
    add_negative_train_samples=False,
)

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, heads=heads, concat=True, edge_dim=1)
        self.conv2 = GATConv(2 * out_channels * heads, out_channels, heads=1, concat=False, edge_dim=1)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        return self.conv2(x, edge_index, edge_attr=edge_attr)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index, edge_attr=train_data.edge_attr)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

graph_list = torch.load(f"GraphData/{opt.dataset}_graphs_{opt.type}.pt")

epochs = 400
embs = []

for idx, data in tqdm(enumerate(graph_list), total=len(graph_list)):
    if getattr(data, "edge_weight", None) is None:
        raise ValueError("Expected data.edge_weight to exist, but it's missing.")
    data.edge_attr = data.edge_weight.view(-1, 1)

    data = data.to(device)

    train_data, val_data, test_data = transform(data)

    for d in (train_data, val_data, test_data):
        if getattr(d, "edge_attr", None) is not None:
            d.edge_attr = d.edge_attr.view(-1, 1).to(device)

    in_channels, out_channels = data.num_features, 8
    model = GAE(GATEncoder(in_channels, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        loss = train()

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, edge_attr=data.edge_attr)
        embs.append(z.cpu())

save_path = f"GraphData/{opt.dataset}_graph_{opt.type}_embs.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(embs, save_path)

# run code
# python autoencoder.py --dataset $dataset --type $type
# e.g. python autoencoder.py --dataset Wildfire --type train