import numpy as np
import random
import copy
import pickle as pk

from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB

from utils import dict_append, del_ext_name, load_pl


class Net:
    def __init__(self):
        self.w = 1
        self.tail = []
        self.head = []


class Hypergraph:
    def __init__(self) -> None:
        """
        self.e2n: 每个edge存储2个列表，self.e2n[0]是tail_node, self.e2n[1]是head_node
        self.n2e: 每个node存储2个列表，self.n2e[0]此结点是tail_node的edge_list, self.e2n[1]此结点时head_node的edge_list
        """
        self.design: str = None
        self.hg_file: str = None
        self.pl_file: str = None
        self.design_pth: str = None
        self.num_edge: int = None
        self.num_node: int = None
        self.edge_width: list = None
        self.e2n: list = None
        self.n2e: list = None
        self.pl: tuple = None

    def read_from_file(self, hg_file):
        print(f"read {hg_file}")
        self.hg_file = hg_file
        self.design = del_ext_name(hg_file)
        self.design_pth = os.path.dirname(hg_file)
        self.load_hypergraph(hg_file)
        self.generate_n2e()

    def build_from_config(self, pl_config, hg_file):
        print(f"generate {hg_file}")
        self.hg_file = hg_file
        self.design = del_ext_name(hg_file)
        self.design_pth = os.path.dirname(hg_file)

        params = Params()
        placedb = PlaceDB()
        params.load(pl_config)
        placedb(params)
        self.read_from_db(placedb)
        self.write(hg_file.replace(".dire", ""))

    def read_from_db(self, placedb: PlaceDB):
        # 遍历 placedb.net2pin_map, placedb.net_weights
        num_node = placedb.num_physical_nodes
        num_edge = placedb.num_nets
        e2n_lst = []
        edge_weight = []
        cnt = 0
        for e, w in zip(placedb.net2pin_map, placedb.net_weights):
            edge_weight.append(int(w))
            node_lst = np.unique([placedb.pin2node_map[p] for p in e]).tolist()
            if len(node_lst) == 1:
                node_lst.append(num_node)
                num_node += 1
                cnt += 1
        self.e2n = e2n_lst
        self.generate_n2e(e2n_lst)
        self.edge_width = edge_weight
        self.num_edge = num_edge
        self.num_node = num_node

    def read_pl(self, pl_file):
        print(f"read {pl_file}")
        self.pl = load_pl(pl_file)

    def write(self, dst=None):
        if dst is None:
            dst = self.hg_file
        print(f"write {dst}")
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
            for l in f:
                l = l.split()
                w = int(l[0])
                tail_lst, head_lst = [], []
                lst = tail_lst
                for nid in l[1:]:
                    nid = int(nid)
                    if nid == -1:
                        lst = head_lst
                    else:
                        lst.append(nid - 1)  # nid - 1 是因为 .hg 文件中的 nid 从 1 开始
                edge_width.append(w)
                e2n_lst.append((tail_lst, head_lst))
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
        for eid, (tail_lst, head_lst) in enumerate(e2n):
            for t in tail_lst:
                if t not in n2e:
                    n2e[t] = [], []
                n2e[t][0].append(eid)
            for h in head_lst:
                if h not in n2e:
                    n2e[h] = [], []
                n2e[h][1].append(eid)

        # 将 dict 转换成 list
        nid_lst = list(n2e.keys())
        nid_lst.sort()
        n2e_lst = []
        for t in nid_lst:
            n2e_lst.append(n2e[t])
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
        for nid, (tail_lst, head_lst) in enumerate(n2e):
            for t in tail_lst:
                if t not in e2n:
                    e2n[t] = [], []
                e2n[t][0].append(nid)
            for h in head_lst:
                if h not in e2n:
                    e2n[h] = [], []
                e2n[h][1].append(eid)

        # 将 dict 转换成 list
        eid_lst = list(e2n.keys())
        eid_lst.sort()
        e2n_lst = []
        for eid in eid_lst:
            e2n_lst.append(e2n[eid])

        # initialize
        self.e2n = e2n_lst
        return e2n_lst

    def find_neighbors(self, nid: int):  # TODO DiHypergraph继承后，能否修改函数参数
        """
        is_forward: True 表示以nid为tail, 寻找其对应的head邻居
        """
        self_idx = 0 if is_forward else 1
        nei_idx = 1 - self_idx
        nei_node = []
        nei_weight = []
        for e in self.n2e[nid][self_idx]:
            w = self.edge_width[e]
            for n2 in self.e2n[e][nei_idx]:
                nei_node.append(n2)
                nei_weight.append(w)
        if len(nei_node) > 0:
            nei_weight = np.bincount(nei_node, weights=nei_weight).astype(int)
            nei_node = np.unique(nei_node)
            nei_weight = nei_weight[nei_node]
        return nei_node, nei_weight


def find_error(lst, w):
    err_lst = []
    for i, (a, b) in enumerate(lst):
        n = len(a) + len(b)
        if n <= w:
            err_lst.append([i, n])
    return err_lst


class DiHypergraph(Hypergraph):
    def __init__(self):
        """
        self.e2n: 每个edge存储2个列表，self.e2n[0]是tail_node, self.e2n[1]是head_node
        self.n2e: 每个node存储2个列表，self.n2e[0]此结点是tail_node的edge_list, self.e2n[1]此结点时head_node的edge_list
        self.hg_src_file: 有向图文件
        self.hg_file: 无向图文件
        """
        self.design: str = None
        self.hg_file: str = None  # 无向图文件
        self.hg_src_file: str = None  # 有向图文件
        self.pl_file: str = None
        self.design_pth: str = None
        self.real_num_node: int = None
        self.num_edge: int = None
        self.num_node: int = None
        self.edge_width: list = None
        self.hpwl: list = None
        self.aver_hpwl: float = None
        self.e2n: list = None
        self.n2e: list = None
        self.pl: tuple = None
        self.vir_edge: list = []

    def build_from_config(self, pl_config, hg_file):
        print(f"generate {hg_file}")
        self.hg_file = hg_file
        self.design = del_ext_name(hg_file)
        self.design_pth = os.path.dirname(hg_file)

        params = Params()
        placedb = PlaceDB()
        params.load(pl_config)
        placedb(params)
        self.read_from_db(placedb)
        self.cal_hpwl()
        self.rebuild_edge_weight()
        self.write_dire(hg_file)
        self.write(hg_file.replace(".dire", ""))

    def read_from_file(self, hg_file):
        print(f"read {hg_file}")
        self.hg_src_file = hg_file
        self.hg_file = hg_file.replace(".dire", "")
        self.design = del_ext_name(hg_file)
        self.design_pth = os.path.dirname(hg_file)
        self.load_hypergraph(hg_file)
        self.generate_n2e()
        self.real_num_node = len(self.pl[0])
        self.cal_hpwl()

    def read_from_db(self, placedb: PlaceDB):
        # 遍历 placedb.net2pin_map, placedb.net_weights
        # TODO 对于只连接了1个点的边，应该如何处理
        # TODO 如果删除只连接了1个点的边，之后产生了没有连接边的点，应该如何处理
        self.real_num_node = placedb.num_physical_nodes
        num_node = placedb.num_physical_nodes
        num_edge = placedb.num_nets
        e2n_lst = []
        edge_width = []
        cnt = 0
        for e, w in zip(placedb.net2pin_map, placedb.net_weights):
            edge_width.append(w)
            tail_lst = []
            head_lst = []
            for p in e:
                if placedb.pin_direct[p].decode() == "INPUT":
                    tail_lst.append(placedb.pin2node_map[p])
                else:
                    head_lst.append(placedb.pin2node_map[p])
            if len(tail_lst) == 0:
                tail_lst.append(num_node)
                num_node += 1
                cnt += 1
            if len(head_lst) == 0:
                head_lst.append(num_node)
                num_node += 1
                cnt += 1
            e2n_lst.append((tail_lst, head_lst))
        edge_width = np.array(edge_width)
        self.e2n = e2n_lst
        self.generate_n2e(e2n_lst)
        self.edge_width = edge_width
        self.num_edge = num_edge
        self.num_node = num_node

    def load_hypergraph(self, hg_file=None):
        """
        从.dire文件中读取超图结构
        """
        if hg_file is None:
            hg_file = self.hg_file
        e2n_list = []
        vir_edge = []
        edge_width = []
        with open(hg_file, encoding="utf-8") as f:
            num_edge, num_node, _ = f.readline().split()
            num_edge, num_node = int(num_edge), int(num_node)
            for l in f:
                l = l.split()
                w = float(l[0])
                if abs(w - (-1)) < 1e-4:
                    break
                tail_lst, head_lst = [], []
                lst = tail_lst
                for nid in l[1:]:
                    nid = int(nid)
                    if nid == -1:
                        lst = head_lst
                    else:
                        lst.append(nid - 1)  # nid - 1 是因为 .hg 文件中的 nid 从 1 开始
                edge_width.append(w)
                e2n_list.append((tail_lst, head_lst))
            for l in f:
                l = l.split()
                vir_edge.append([float(l[0]), int(l[1]) - 1, int(l[3]) - 1])
        # initialize
        edge_width = np.array(edge_width, dtype=np.float64)
        self.num_edge = num_edge
        self.num_node = num_node
        self.edge_width = edge_width
        self.e2n = e2n_list
        self.vir_edge = vir_edge
        return num_edge, num_node, edge_width, e2n_list

    def read_pl(self, pl_file):
        print(f"read {pl_file}")
        self.pl = load_pl(pl_file)

    def cal_hpwl_e(self, eid):
        pos_x, pos_y = self.pl
        tail_list, head_list = self.e2n[eid]
        node_list = tail_list + head_list
        xmax = ymax = 0
        xmin = ymin = 1e10
        for n in node_list:
            if n >= self.real_num_node:
                return np.inf
            if pos_x[n] > xmax:
                xmax = pos_x[n]
            if pos_x[n] < xmin:
                xmin = pos_x[n]
            if pos_y[n] > ymax:
                ymax = pos_y[n]
            if pos_y[n] < ymin:
                ymin = pos_y[n]
        hpwl = xmax - xmin + ymax - ymin
        return hpwl

    def cal_hpwl(self):
        self.hpwl = np.zeros(self.num_edge, dtype=np.float64)
        for eid in range(self.num_edge):
            self.hpwl[eid] = self.cal_hpwl_e(eid)
        self.aver_hpwl = np.average(self.hpwl[self.hpwl != np.inf])

    def rebuild_edge_weight(self):
        # TODO 根据实验得知，添加虚拟边后，切分了大量HPWL较小的边
        # TODO 所以考虑将HPWL的信息结合进边的权重中，得到更准确的描述
        # TODO 希望被切的边是
        # TODO 1. 权重小
        # TODO 2. HPWL大
        # TODO 实际会切分权重小的边
        # TODO 目前暂定混合权重为 weight / HPWL * aver(HPWL)
        self.edge_width = self.edge_width * self.aver_hpwl * 100 / self.hpwl

    def write(self, dst=None):
        if dst is None:
            dst = self.hg_file
        print(f"write {dst}")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(f"{self.num_edge} {self.num_node} 1\n")  # 1 表示加权图
            s = ""
            for w, (tail_lst, head_lst) in zip(self.edge_width, self.e2n):
                s += str(int(w))
                for nid in tail_lst + head_lst:
                    s += " " + str(nid + 1)  # nid 需从 1 开始
                s += "\n"
            for w, t, h in self.vir_edge:
                s += f"{int(w)} {t+1} {h+1}\n"
            f.write(s)
        self.hg_file = dst

    def write_dire(self, dst=None):
        if dst is None:
            dst = self.hg_file
        print(f"write {dst}")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(f"{self.num_edge} {self.num_node} 1\n")  # 1 表示加权图
            s = ""
            for w, (tail_lst, head_lst) in zip(self.edge_width, self.e2n):
                s += str(int(w))
                for nid in tail_lst:
                    s += " " + str(nid + 1)  # nid 需从 1 开始
                s += " -1"
                for nid in head_lst:
                    s += " " + str(nid + 1)  # nid 需从 1 开始
                s += "\n"
            s += "-1\n"
            for w, t, h in self.vir_edge:
                s += f"{int(w)} {t+1} -1 {h+1}\n"
            f.write(s)
        self.hg_src_file = dst

    def generate_n2e(self, e2n=None):
        """
        根据 e2n 生成 n2e
        """
        if e2n is None:
            e2n = self.e2n
        # cnt = np.zeros(self.num_node)
        n2e = dict()
        for eid, (tail_lst, head_lst) in enumerate(e2n):
            for t in tail_lst:
                if t not in n2e:
                    n2e[t] = [], []
                n2e[t][0].append(eid)
                # cnt[t] += 1
            for h in head_lst:
                if h not in n2e:
                    n2e[h] = [], []
                n2e[h][1].append(eid)
                # cnt[h] += 1
        # for i, x in enumerate(cnt):
        #     if x == 0:
        #         print(i, end=" ")

        # 将 dict 转换成 list
        nid_lst = list(n2e.keys())
        nid_lst.sort()
        n2e_lst = []
        for t in nid_lst:
            n2e_lst.append(n2e[t])
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
        for nid, (tail_lst, head_lst) in enumerate(n2e):
            for t in tail_lst:
                if t not in e2n:
                    e2n[t] = [], []
                e2n[t][0].append(nid)
            for h in head_lst:
                if h not in e2n:
                    e2n[h] = [], []
                e2n[h][1].append(nid)

        # 将 dict 转换成 list
        eid_lst = list(e2n.keys())
        eid_lst.sort()
        e2n_lst = []
        for eid in eid_lst:
            e2n_lst.append(e2n[eid])

        # initialize
        self.e2n = e2n_lst
        return e2n_lst

    def find_neighbors(self, nid: int, is_forward):
        """
        is_forward: True 表示以nid为tail, 寻找其对应的head邻居
        """
        self_idx = 0 if is_forward else 1
        nei_idx = 1 - self_idx
        nei_node = []
        nei_weight = []
        for e in self.n2e[nid][self_idx]:
            w = self.edge_width[e]
            for n2 in self.e2n[e][nei_idx]:
                nei_node.append(n2)
                nei_weight.append(w)
        if len(nei_node) > 0:
            nei_weight = np.bincount(nei_node, weights=nei_weight).astype(int)
            nei_node = np.unique(nei_node)
            nei_weight = nei_weight[nei_node]
        return nei_node, nei_weight

    def _cal_dataflow_one_dire(self, nid, k, w_thre, is_forward):
        # TODO 数据流具体计算方法还需修改
        n1_flow = {nid: 0}
        one_loop_nei = set([nid])  # TODO 是否对一阶邻居添加虚拟边？重复了？
        q = [nid]
        for i in range(k):
            next_neighbors = []  # 下次循环需要访问邻居的结点
            for n_tmp in q:  # 遍历第i阶邻居
                nei_nodes, nei_weight = self.find_neighbors(n_tmp, is_forward)
                if n_tmp == nid:  # 一阶邻居
                    one_loop_nei.update(nei_nodes)
                tmp_nei = []  # n_tmp 的邻居中，下次循环需要访问邻居的结点
                for n2, w in zip(nei_nodes, nei_weight):
                    w = w / 2**i + n1_flow[n_tmp] / 2  # TODO 是否加上 tmp_nei 的权重，感觉是必要的
                    if n2 in n1_flow:  # 此处不将n2添加到tmp_nei中，因为之前访问过n2了
                        w += n1_flow[n2]  # w_self
                        n1_flow[n2] = w
                    elif w >= w_thre:  # w < w_thre 时，忽略，不再访问其邻居
                        tmp_nei.append(n2)
                        n1_flow[n2] = w
                next_neighbors += tmp_nei
            q = next_neighbors
        vir_edge = []
        for n2, w in n1_flow.items():
            if n2 not in one_loop_nei:
                if is_forward:
                    vir_edge.append([w, nid, n2])
                else:
                    vir_edge.append([w, n2, nid])
        return vir_edge

    def cal_dataflow(self, k=3, w_thre=5):
        """
        k: 到k阶邻居为止
        w_thre: 小于 w_thre 的 width，不再继续寻找其邻居
        """
        # 每个点寻找其邻居
        # vir_edge = [ (dataflow, n1, n2), ...], n1: tail, n2: head
        vir_edge = []
        for n1 in range(self.num_node):
            vir_edge.extend(self._cal_dataflow_one_dire(n1, k, w_thre, True))  # 向前找 k 阶邻居, 此时n1是tail
            vir_edge.extend(self._cal_dataflow_one_dire(n1, k, w_thre, False))  # 向后找 k 阶邻居, 此时n1是head
        return vir_edge

    def _cal_dataflow_one_dire2(self, nid, k, w_thre, is_forward):
        """放宽约束"""
        # TODO 数据流具体计算方法还需修改
        n1_flow = {nid: 0}
        one_loop_nei = set([nid])  # TODO 是否对一阶邻居添加虚拟边？重复了？
        q = [nid]
        for i in range(k):
            next_neighbors = []  # 下次循环需要访问邻居的结点
            for n_tmp in q:  # 遍历第i阶邻居
                nei_nodes, nei_weight = self.find_neighbors(n_tmp, is_forward)
                if n_tmp == nid:  # 一阶邻居
                    one_loop_nei.update(nei_nodes)
                tmp_nei = []  # n_tmp 的邻居中，下次循环需要访问邻居的结点
                for n2, w in zip(nei_nodes, nei_weight):
                    w = w / 2**i + n1_flow[n_tmp] / 2  # TODO 是否加上 tmp_nei 的权重，感觉是必要的
                    if n2 in n1_flow:  # 此处不将n2添加到tmp_nei中，因为之前访问过n2了
                        w += n1_flow[n2]  # w_self
                        n1_flow[n2] = w
                    else:
                        tmp_nei.append(n2)
                        n1_flow[n2] = w
                next_neighbors += tmp_nei
            q = next_neighbors
        vir_edge = []
        for n2, w in n1_flow.items():
            if w >= w_thre and n2 not in one_loop_nei:
                if is_forward:
                    vir_edge.append([w, nid, n2])
                else:
                    vir_edge.append([w, n2, nid])
        return vir_edge

    def cal_dataflow2(self, k=3, w_thre=5):
        """
        放宽约束
        k: 到k阶邻居为止
        w_thre: 小于 w_thre 的 width，不再继续寻找其邻居
        """
        # 每个点寻找其邻居
        # vir_edge = [ (dataflow, n1, n2), ...], n1: tail, n2: head
        vir_edge = dict()
        for n1 in range(self.num_node):
            new_edge_tail = self._cal_dataflow_one_dire2(n1, k, w_thre, True)
            new_edge_head = self._cal_dataflow_one_dire2(n1, k, w_thre, False)
            vir_edge.update(
                {
                    (t, h): w
                    for w, t, h in new_edge_tail + new_edge_head
                    if not ((t, h) in vir_edge and w < vir_edge[(t, h)])
                }
            )
        vir_edge = [[w, t, h] for (t, h), w in vir_edge.items()]
        return vir_edge

    def add_vir_edge(self, vir_edge: list):
        # for i, (w, n1, n2) in enumerate(vir_edge):
        #     self.edge_width.append(w)
        #     self.e2n.append(([n1], [n2]))
        #     eid = i + self.num_edge
        #     self.n2e[n1][0].append(eid)
        #     self.n2e[n2][1].append(eid)
        self.vir_edge = copy.deepcopy(vir_edge)
        self.num_edge += len(vir_edge)

    def dataflow_improve(self, m_type, new_dataflow=False):
        """
        new_dataflow: 是否不使用历史记录，重新计算
        """
        print(f"dataflow_improve {self.hg_file}")
        vir_edge_file = os.path.join(
            self.design_pth, "vir_edge.bin" if m_type == "base" else f"vir_edge.{m_type}.bin"
        )
        if os.path.exists(vir_edge_file) and not new_dataflow:
            with open(vir_edge_file, "rb") as f:
                vir_edge = pk.load(f)
        else:
            vir_edge = self.cal_dataflow2(k=3)
            with open(vir_edge_file, "wb") as f:
                pk.dump(vir_edge, f)
        self.add_vir_edge(vir_edge)
        vir_file = self.hg_src_file.replace(".hg", f".{m_type}.vir")
        self.write_dire(vir_file)
        self.write(vir_file.replace(".dire", ""))


import glob
import os


def generate_single_hg_file(src, dst):
    params = Params.Params()
    placedb = PlaceDB.PlaceDB()
    params.load(src)
    placedb(params)

    hg = DiHypergraph()
    hg.read_from_db(placedb)
    hg.write(dst)
    return hg


def read_benchmark(benchmark):
    hg_file_lst = glob.glob(os.path.join("res", benchmark, "hypergraph", "adaptec*.hg"))
    hg_file_lst.sort()

    hg_lst = []
    for hg_file in hg_file_lst:
        print(hg_file)
        hg = DiHypergraph()
        hg.read_from_file(hg_file)
        hg_lst.append(hg)
    return hg_lst


def all_max_edge(hg_lst: list[DiHypergraph]):
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


def add_all_vir_edge(hg_lst: list[DiHypergraph]):
    for hg in hg_lst:
        print(hg.hg_file)
        print("start calculate dataflow")
        vir_edge = hg.cal_dataflow()
        print("start add virtual edge")
        hg.add_vir_edge(vir_edge)
        dst_file = hg.hg_file + ".vir"
        print("start write file\n")
        hg.write(dst_file)


def check(n1, n2, hg: DiHypergraph):  #! 此函数在当前改为有向超图后，已无效
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
    import jstyleson

    benchmark = "simple"
    b_pth = os.path.join("res", benchmark)
    os.system(f"mkdir -p {b_pth}")
    config_file = os.path.join("par_config", benchmark, "config.json")
    with open(config_file, encoding="utf-8") as f:
        config = jstyleson.load(f)

    design_lst = config["design"]
    for design in design_lst:
        design_pth = os.path.join(b_pth, design)
        os.system(f"mkdir -p {design_pth}")
        hg = DiHypergraph()
        hg_file = os.path.join(design_pth, design + ".vir" + ".dire")
        hg_ori_file = os.path.join(design_pth, design + ".hg" + ".dire")
        hg.read_from_file(hg_ori_file)
        hg.dataflow_improve()
