import torch
import time
import math
import dgl
import numpy as np
import torch.nn as nn
from dgl.data import citation_graph as citegrh
from dgl import DGLGraph
import networkx as nx
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class GCN1(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout) -> None:
        super(GCN1, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def gcn_msg(edge):
    msg = edge.src["h"] * edge.src["norm"]
    return {"m": msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox["m"], dim=1) * node.data["norm"]
    return {"h": accum}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data["h"]
        if self.bias is not None:
            h += self.bias
        if self.activation:
            h = self.activation(h)
        return {"h": h}


class GCNLayer2(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True) -> None:
        super(GCNLayer2, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.weight.size())
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata["h"] = torch.mm(h, self.weight)
        self.g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = self.g.ndata.pop("h")
        return h


class GCN2(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout) -> None:
        super(GCN2, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer2(g, in_feats, n_hidden, activation, dropout))
        for i in range(n_layers - 2):
            self.layers.append(GCNLayer2(g, n_hidden, n_hidden, activation, dropout))
        self.layers.append(GCNLayer2(g, n_hidden, n_classes, None, dropout))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h


class GCNLayer3(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer3, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        self.g.ndata["h"] = self.g.ndata["norm"] * torch.mm(h, self.weight)
        self.g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        h = self.g.ndata.pop("h") * self.g.ndata["norm"]
        if self.bias:
            h += self.bias
        if self.activation:
            h = self.activation(h)
        return h


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


if __name__ == "__main__":
    torch.manual_seed(0)
    dropout = 0.5
    gpu = -1
    lr = 0.01
    n_epochs = 200
    n_hidden = 16
    n_layers = 2
    weight_decay = 5e-4
    self_loop = True

    dataset = citegrh.load_cora("./dataset")
    g = dataset[0]
    features = torch.FloatTensor(g.ndata["feat"])
    labels = torch.LongTensor(g.ndata["label"])
    train_mask = torch.BoolTensor(g.ndata["train_mask"])
    val_mask = torch.BoolTensor(g.ndata["val_mask"])
    test_mask = torch.BoolTensor(g.ndata["test_mask"])

    in_feats = features.shape[1]
    n_classes = dataset.num_classes
    n_edges = g.num_edges()

    g: nx.Graph = g.to_networkx()
    if self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = dgl.from_networkx(g)

    if gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata["norm"] = norm.unsqueeze(1)

    model = GCN1(g, in_feats, n_hidden, n_classes, n_layers, F.relu, dropout)
    if cuda:
        model.cuda()
    loss_fn = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    dur = []
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()

        logits = model(features)
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        if epoch % 10 == 0:
            acc = evaluate(model, features, labels, val_mask)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | ETputs(KTEPS) {:.2f}".format(
                    epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000
                )
            )
    acc = evaluate(model, features, labels, test_mask)
    print(f"Test accuracy {acc:.2f}")
