import torch
from torch import optim
import torch.nn.functional as F
import copy
import numpy as np
import random

from dhg import Hypergraph
from dhg.models import HGNN

from hypergraph import DiHypergraph
from utils import dict_append

import matplotlib
matplotlib.use("TKAgg")

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
    
def generate_par(node_k, pos_x, pos_y):
    par_dict = dict()
    # 先为每个类构建array(id)
    for _id, i in enumerate(node_k):
        dict_append(par_dict, i, _id)
    # 为每一个类构建[n, array(id), array(x), array(y)]
    for i, id_list in par_dict.items():
        id_list = np.array(id_list)
        x = pos_x[id_list]
        y = pos_y[id_list]
        nk = len(id_list)
        par_dict[i] = [nk, id_list, x, y]
    return par_dict


def cal_loss(p_mat, hg, k):
    pos_x, pos_y = hg.pl
    pos_x = torch.tensor(pos_x, requires_grad=True)
    pos_y = torch.tensor(pos_y, requires_grad=True)
    p_mat = F.softmax(p_mat[:, :k], dim=1)
    node_k = p_mat.argmax(dim=1)

    loss = torch.tensor(0, requires_grad=True)
    for i in range(k):
        idx = node_k == i
        x = pos_x[idx]
        y = pos_y[idx]
        loss += torch.sqrt(torch.var(x) + torch.var(y))
    return loss


def train(model, X, hg, train_idx, optimizer, k):
    model.train()
    optimizer.zero_grad()
    outs = model(X, hg)
    outs = outs[train_idx]
    loss = cal_loss(outs, hg, k)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def infer(net, X, hg, idx, k):
    net.eval()
    outs = net(X, hg)
    outs = outs[idx]
    res = cal_loss(outs, k)
    return res.item()


device = torch.device("cuda")

num_feat = 1024
k = 1024

hg = DiHypergraph()#TODO
X = torch.randn(hg.num_node, num_feat)
G = DiHypergraph(hg.num_node, hg.e2n, hg.edge_width)

train_mask = None#TODO
val_mask = None#TODO
test_mask = None#TODO

model = HGNN(X.shape[1], 1024, k, use_bn=True)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

X = X.to(device)
G = G.to(device)
model = model.to(device)

best_state = None
best_epoch, best_val = 0, 0
for epoch in range(200):
    train(model, X, hg, train_mask, optimizer, k)
    if epoch % 1 == 0:
        with torch.no_grad():
            val_res = infer(model, X, hg, val_mask, k)
        if val_res < best_val:
            print(f"update best: {val_res:.5f}")
            best_epoch = epoch
            best_val = val_res
            best_state = copy.deepcopy(model.state_dict())

print("\ntrain finished!")
print(f"best val: {best_val:.5f}")
print("test...")
model.load_state_dict(best_state)
res = infer(model, X, G, test_mask, k)
print(f"final result: epoch: {best_epoch}, val: {res:.5f}")
