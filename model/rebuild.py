import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import tqdm
import torchmetrics.functional.classification as MFC

from torch import optim
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler, DataLoader

# from dgl.data import Da


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.dropout = nn.Dropout(p=0.5)
        self.layers = nn.ModuleList(
            [
                dglnn.GraphConv(in_size, hid_size, activation=F.relu, allow_zero_in_degree=True),
                dglnn.GraphConv(hid_size, hid_size, activation=F.relu, allow_zero_in_degree=True),
                dglnn.GraphConv(hid_size, out_size, allow_zero_in_degree=True),
            ]
        )

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != 0:
                h = self.dropout(h)
            h = layer(block, h)
        return h

    # def inference(self, g: dgl.DGLGraph, device, batch_size, _feat):
    #     sampler = MultiLayerFullNeighborSampler(3, prefetch_node_feats=[_feat])
    #     dataloader = DataLoader(
    #         g,
    #         torch.tensor(g.num_nodes()).to(g.device),
    #         sampler,
    #         device,
    #         batch_size=batch_size,
    #         drop_last=False,
    #         shuffle=False,
    #     )

    #     feat = g.ndata[_feat].to(device)
    #     res = torch.zeros(g.num_nodes(), self.out_size).to(device)
    #     with torch.no_grad():
    #         for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
    #             x = feat[input_nodes]
    #             y_hat = self.forward(blocks, x)
    #             res[output_nodes] = y_hat
    #     return res


def train_one_epoch(model, data_loader, optimizer):
    model.train()
    epoch_loss = 0
    for n, (input_nodes, output_nodes, blocks) in enumerate(data_loader):
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / (n + 1)


def evaluate(model, dataloader, num_classes):
    model.eval()
    y = []
    y_hat = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            x = blocks[0].srcdata["feat"]
            yi = blocks[-1].dstdata["label"]
            yi_hat = model(blocks, x)
            y.append(yi)
            y_hat.append(yi_hat)
        y = torch.cat(y).to("cuda")
        y_hat = torch.cat(y_hat).to("cuda")
        acc = MFC.multiclass_accuracy(y_hat, y, num_classes, "weighted")
    return acc.item()


def inference(g: dgl.DGLGraph, model, idx_lst, batch_size, num_classes):
    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"])
    dataloader = DataLoader(
        g, torch.arange(g.num_nodes()), sampler, device="cuda", batch_size=batch_size, use_uva=True
    )
    model.eval()
    res = torch.zeros(g.num_nodes(), num_classes, device="cuda")
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            x = blocks[0].srcdata["feat"]
            y_hat = model(blocks, x)
            res[output_nodes] = y_hat
        res = res[idx_lst]
        y = g.ndata["label"][idx_lst].to("cuda")
        acc = MFC.multiclass_accuracy(res, y, num_classes, "weighted")
    return acc


def train(g, model, dataset, _feat):
    train_idx = dataset.train_idx.to("cuda")
    val_idx = dataset.val_idx.to("cuda")
    sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=[_feat])
    train_dataloader = DataLoader(
        g, train_idx, sampler, "cuda", batch_size=2048, shuffle=True, drop_last=False, use_uva=True
    )
    val_dataloader = DataLoader(
        g, val_idx, sampler, "cuda", batch_size=2048, shuffle=True, drop_last=False, use_uva=True
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for epoch in range(10):
        epoch_loss = train_one_epoch(model, train_dataloader, optimizer)
        acc = evaluate(model, val_dataloader, dataset.num_classes)
        print(f"Epoch {epoch:5d} | Loss {epoch_loss:.4f} | Accuracy {acc:.4f}")


if __name__ == "__main__":
    from dgl.data import AsNodePredDataset
    from ogb.nodeproppred import DglNodePropPredDataset

    print("Loading data...")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]

    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 512, out_size).to("cuda")

    print("Training")
    train(g, model, dataset, "feat")

    print("Testing")
    acc = inference(g, model, dataset.test_idx, 2048, dataset.num_classes)
    print(f"Test Accuracy {acc.item():.4f}")
