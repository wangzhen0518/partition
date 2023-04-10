import numpy as np

from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB

from utils import dict_append, del_ext_name

pl_ext = "gp"  # gp or ntup


class hypergraph:
    def __init__(self):
        self.design: str = None
        self.hg_file: str = None
        self.pl_file: str = None
        self.design_pth: str = None
        self.num_edge: int = None
        self.num_node: int = None
        self.edge_width: list = None
        self.e2n: list = None
        self.n2e: list = None

    def read_from_file(self, hg_file):
        self.hg_file = hg_file
        self.design = del_ext_name(hg_file)
        self.design_pth = os.path.dirname(hg_file)
        self.pl_file = os.path.join(self.design_pth, self.design + f".{pl_ext}.pl")
        self.load_hypergraph(hg_file)
        self.generate_n2e()

    def build_from_config(self, pl_config, hg_file):
        self.hg_file = hg_file
        self.design = del_ext_name(hg_file)
        self.design_pth = os.path.dirname(hg_file)
        self.pl_file = os.path.join(self.design_pth, self.design + f".{pl_ext}.pl")

        params = Params()
        placedb = PlaceDB()
        params.load(pl_config)
        placedb(params)
        self.read_from_db(placedb)
        self.write(hg_file)

    def read_from_db(self, placedb):
        # 遍历 placedb.net2pin_map, placedb.net_weights
        e2n_lst = []
        edge_weight = []
        for e, w in zip(placedb.net2pin_map, placedb.net_weights):
            node_lst = np.unique([placedb.pin2node_map[p] for p in e]).tolist()
            if len(node_lst) > 1:
                e2n_lst.append(node_lst)
                edge_weight.append(int(w))
        self.edge_width = edge_weight
        self.generate_n2e(e2n_lst)
        self.generate_e2n()  # 重新生成一遍，以删除没有连边的结点，并调整所有结点的编号
        self.num_edge = len(self.e2n)
        self.num_node = len(self.n2e)

    def write(self, dst=None):
        if dst is None:
            dst = self.hg_file
        with open(dst, "w", encoding="utf-8") as f:
            f.write(f"{self.num_edge} {self.num_node} 1\n")  # 1 表示加权图
            for w, node_lst in zip(self.edge_width, self.e2n):
                s = str(w)
                for nid in node_lst:
                    s += " " + str(nid + 1)  # nid 需从 1 开始
                s += "\n"
                f.write(s)
        self.hg_file = dst

    def load_hypergraph(self, hg_file=None):
        if hg_file is None:
            hg_file = self.hg_file
        e2n_lst = []
        edge_width = []
        with open(hg_file, encoding="utf-8") as f:
            num_edge, num_node, _ = f.readline().split()
            num_edge, num_node = int(num_edge), int(num_node)
            for l in f:
                l = l.split()
                w = int(l[0])
                node_lst = [int(nid) - 1 for nid in l[1:]]  # nid - 1 是因为 .hg 文件中的 nid 从 1 开始
                edge_width.append(w)
                e2n_lst.append(node_lst)
        # initialize
        self.num_edge = num_edge
        self.num_node = num_node
        self.edge_width = edge_width
        self.e2n = e2n_lst
        return num_edge, num_node, edge_width, e2n_lst

    def generate_n2e(self, e2n=None):
        """
        根据 e2n 生成 n2e
        """
        if e2n is None:
            e2n = self.e2n

        n2e = dict()
        for eid, node_lst in enumerate(e2n):
            for nid in node_lst:
                dict_append(n2e, nid, eid)

        # 将 dict 转换成 list，同时除去了没有连边的点
        nid_lst = list(n2e.keys())
        nid_lst.sort()
        n2e_lst = []
        for nid in nid_lst:
            n2e_lst.append(n2e[nid])
        # initialize
        self.n2e = n2e_lst
        return n2e_lst

    def generate_e2n(self, n2e=None):
        """
        根据 n2e 生成 e2n
        """
        if n2e is None:
            n2e = self.n2e

        e2n = dict()
        for nid, edge_lst in enumerate(n2e):
            for eid in edge_lst:
                dict_append(e2n, eid, nid)

        # 将 dict 转换成 list, 同时除去了没有连点的边
        eid_lst = list(e2n.keys())
        eid_lst.sort()
        e2n_lst = []
        for eid in eid_lst:
            e2n_lst.append(e2n[eid])

        # initialize
        self.e2n = e2n_lst
        return e2n_lst

    def find_neighbors(self, nid: int):
        nei_nodes = []
        nei_weight = []
        for e in self.n2e[nid]:
            w = self.edge_width[e]
            for n2 in self.e2n[e]:
                if n2 != nid:
                    nei_nodes.append(n2)
                    nei_weight.append(w)
        nei_weight = np.bincount(nei_nodes, weights=nei_weight).astype(int)
        nei_nodes = np.unique(nei_nodes)
        nei_weight = nei_weight[nei_nodes]
        return nei_nodes, nei_weight

    def cal_dataflow(self, k=3, w_thre=5):
        """
        k: 到k阶邻居为止
        w_thre: 小于 w_thre 的 width，不再继续寻找其邻居
        """
        # TODO 数据流具体计算方法还需修改
        # 每个点寻找其邻居
        # vir_edge = [ (dataflow, n1, n2), ...], n2 > n1
        vir_edge = []
        for n1 in range(self.num_node):
            n1_flow = dict()
            q = [n1]
            one_loop_nei = set()
            # TODO 是否对一阶邻居添加虚拟边？重复了？
            for i in range(k):
                next_neighbors = []  # 下次循环需要访问邻居的结点
                for n_tmp in q:  # 遍历第i阶邻居
                    nei_nodes, nei_weight = self.find_neighbors(n_tmp)
                    if n_tmp == n1:
                        one_loop_nei.update(nei_nodes)
                    tmp_nei = []  # n_tmp 的邻居中，下次循环需要访问邻居的结点
                    for n2, w in zip(nei_nodes, nei_weight):
                        if n2 > n1:  # TODO去重，是否要移到find_neighbors中
                            if n2 in n1_flow:
                                # 此处不将n2添加到tmp_nei中，因为之前访问过n2了
                                w = (
                                    n1_flow[n2] + (w + n1_flow[n_tmp]) / 2**i
                                )  # TODO 是否加上 tmp_nei 的权重，感觉是必要的
                            elif w >= w_thre:
                                n1_flow[n2] = w
                                tmp_nei.append(n2)
                            # w < w_thre 时，忽略，不再访问其邻居
                    next_neighbors += tmp_nei
                q = next_neighbors
            for n2, w in n1_flow.items():
                if n2 not in one_loop_nei:
                    vir_edge.append((int(w), n1, n2))
        return vir_edge

    def add_vir_edge(self, vir_edge: list):
        for i, (w, n1, n2) in enumerate(vir_edge):
            self.edge_width.append(w)
            self.e2n.append([n1, n2])
            eid = i + self.num_edge
            self.n2e[n1].append(eid)
            self.n2e[n2].append(eid)
        self.num_edge += len(vir_edge)

    def dataflow_improve(self):
        vir_edge = self.cal_dataflow()
        self.add_vir_edge(vir_edge)
        vhg_file = self.hg_file.replace(".hg", ".vir")
        self.write(vhg_file)


import glob
import os


def generate_single_hg_file(src, dst):
    params = Params.Params()
    placedb = PlaceDB.PlaceDB()
    params.load(src)
    placedb(params)

    hg = hypergraph()
    hg.read_from_db(placedb)
    hg.write(dst)
    return hg


def read_benchmark(benchmark):
    hg_file_lst = glob.glob(os.path.join("res", benchmark, "hypergraph", "adaptec*.hg"))
    hg_file_lst.sort()

    hg_lst = []
    for hg_file in hg_file_lst:
        print(hg_file)
        hg = hypergraph()
        hg.read_from_file(hg_file)
        hg_lst.append(hg)
    return hg_lst


def all_max_edge(hg_lst: list[hypergraph]):
    cnt_lst = []
    pos_lst = []
    for hg in hg_lst:
        max_cnt = 1
        max_pos = None
        for n1 in range(hg.num_node):
            nei_nodes, nei_weight = hg.find_neighbors(n1)
            # TODO 有没有办法同时返回最大值及其索引？numpy好像没有，torch有
            cnt = np.max(nei_weight)
            n2 = nei_nodes[np.argmax(nei_nodes)]
            if n2 > n1 and cnt > max_cnt:
                max_cnt = cnt
                max_pos = (n1, n2, max_cnt)
        cnt_lst.append(max_cnt)
        pos_lst.append(max_pos)
        pos = (max_pos[0] + 1, max_pos[1] + 1, max_pos[2])
        print(hg.hg_file, pos)

    with open("max_edge.txt", "w", encoding="utf-8") as f:
        for hg, pos in zip(hg_lst, pos_lst):
            # nid 从 1 开始
            pos = (pos[0] + 1, pos[1] + 1, pos[2])
            f.write(f"{hg.hg_file} {pos}\n")


def add_all_vir_edge(hg_lst: list[hypergraph]):
    for hg in hg_lst:
        print(hg.hg_file)
        print("start calculate dataflow")
        vir_edge = hg.cal_dataflow()
        print("start add virtual edge")
        hg.add_vir_edge(vir_edge)
        dst_file = hg.hg_file + ".vir"
        print("start write file\n")
        hg.write(dst_file)


def check(n1, n2, hg: hypergraph):
    cnt = 0
    edge_lst = []
    for eid, node_lst in enumerate(hg.e2n):
        has_n1 = False
        has_n2 = False
        for nid in node_lst:
            if nid == n1:
                has_n1 = True
            if nid == n2:
                has_n2 = True
        if has_n1 and has_n2:
            cnt += 1
            edge_lst.append(eid)
    return cnt, edge_lst


if __name__ == "__main__":
    # hg = hypergraph()
    # hg.read_from_file("res/ispd2005/hypergraph/adaptec1.hg")
    # all_max_edge([hg])
    hg_lst = read_benchmark("ispd2005")
    print("\nstart all max edge")
    # all_max_edge(hg_lst)
    print("\nstart add virtual edge")
    add_all_vir_edge(hg_lst)
