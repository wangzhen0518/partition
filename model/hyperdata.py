import os
import torch
import dgl
import dgl.data as dgldata
import random
from dgl.data import DGLDataset, CSVDataset, KarateClubDataset, QM7bDataset
import networkx as nx
import sys

sys.path.append("..")

# from ..hypergraph import hypergraph
# from ..utils import load_par

from hypergraph import DiHypergraph  # TODO 包引用，解决路径问题
from utils import load_par  # TODO


class HypergraphDataset(DGLDataset):
    def __init__(
        self,
        name: str,
        hg_list: list[DiHypergraph],
        par_file_list: list[str],
        multiprocess=1,
        raw_dir="../data",
        save_dir="../data",
        verbose=False
        # force_reload=False,
    ):
        self.hg_list = hg_list
        self.par_file_list = par_file_list
        self.multiprocess = multiprocess
        self.graphs = []
        self.labels = []
        self.graph_names = [hg.design for hg in hg_list]
        super(HypergraphDataset, self).__init__(name, raw_dir=raw_dir, save_dir=save_dir, verbose=verbose)

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        print("process")
        self.graphs, self.graph_names = self.generate_graphs()
        self.generate_mask()  # TODO train, valid, test mask
        self.labels = self.generate_labels()

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        assert idx < len(self.graphs), "HypergraphLoader: index out of bounds"
        return self.graphs[idx]

    def __len__(self):
        # 数据样本的数量
        return len(self.graphs)

    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        print(f"save {graph_path}")
        dgldata.save_graphs(str(graph_path), self.graphs, {"labels": self.labels})

    def load(self):
        # 从 `self.save_path` 导入处理后的数据
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        print(f"load {graph_path}")
        graphs, label_dict = dgldata.load_graphs(graph_path)
        self.graphs = graphs
        self.labels = label_dict["labels"]

    def has_cache(self):
        # 检查在 `self.save_path` 中是否存有处理后的数据
        graph_path = os.path.join(self.save_path, self.save_name + ".bin")
        return os.path.exists(graph_path)

    @property
    def num_partition(self):
        return 2

    @property
    def feature_length(self):
        return 128

    @property
    def save_name(self):
        return self.name + f"_{self.num_partition}_{self.feature_length}_dgl_hypergraph"

    def _generate_one_graph(self, hg):
        print(f"generate {hg.design}")
        # g = nx.DiGraph()
        g = dgl.DGLGraph()
        edge_dict = dict()
        for n1 in range(hg.num_node):
            nei_node, nei_weight = hg.find_neighbors(n1, True)
            for n2, w in zip(nei_node, nei_weight):
                edge_dict[(n1, n2)] = w
            nei_node, nei_weight = hg.find_neighbors(n1, False)
            for n2, w in zip(nei_node, nei_weight):
                edge_dict[(n2, n1)] = w
        h, t, w = [], [], []
        for (n1, n2), wi in edge_dict.items():
            h.append(n1)
            t.append(n2)
            w.append(wi)
        g.add_edges(h, t, {"w": torch.FloatTensor(w)})
        g = g.int()

        # # load label
        # par = load_par(par_file)
        # g.ndata['label']=torch.IntTensor(par)
        return g, hg.design

    def generate_graphs(self):
        print("generate graphs")
        graphs = []
        graph_names = []
        res_list = []
        if self.multiprocess > 1:
            from multiprocessing import Pool

            pool = Pool(8)
            for hg in self.hg_list:
                res_list.append(pool.apply_async(self._generate_one_graph, args=(hg,)))
            pool.close()
            pool.join()
            res_list = [res.get() for res in res_list]
        else:
            for hg in self.hg_list:
                res_list.append(self._generate_one_graph(hg))

        for res in res_list:
            g, gname = res
            graphs.append(g)
            graph_names.append(gname)
        return graphs, graph_names

    def generate_labels(self):
        print(f"generate labels")
        labels = []
        for par_file in self.par_file_list:
            par = load_par(par_file)
            labels.append(torch.IntTensor(par))
        labels = torch.vstack(labels)
        return labels

    def generate_mask(self):
        prev_state = random.getstate()
        torch.manual_seed(3407)
        for gi in self.graphs:
            gi: dgl.DGLGraph
            n = gi.num_nodes()
            train_mask = torch.zeros(n, dtype=torch.bool)
            valid_mask = torch.zeros(n, dtype=torch.bool)
            test_mask = torch.zeros(n, dtype=torch.bool)
            all_node = set(range(n))
            train_idx = random.sample(all_node, int(n * 0.7))
            train_mask[train_idx] = True
            all_node.difference_update(train_idx)
            valid_idx = random.sample(all_node, int(n * 0.2))
            valid_mask[valid_idx] = True
            all_node.difference_update(valid_idx)
            test_idx = list(all_node)
            test_mask[test_idx] = True
            gi.ndata["train_mask"] = train_mask
            gi.ndata["valid_mask"] = valid_mask
            gi.ndata["test_mask"] = test_mask
        random.setstate(prev_state)
