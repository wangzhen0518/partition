import numpy as np
import jstyleson
import matplotlib.pyplot as plt
import os

from utils import load_pl, load_par, dict_append, del_ext_name, generate_benchmark_dict

"""
partition的数据结构为
{
    ...,
    k: [n, array(id), array(xi), array(yi)],
    ...
}

n: 第k个类中点的数量
"""


def generate_par(par_list, pl):
    pos_x, pos_y = pl
    pos_x, pos_y = np.array(pos_x), np.array(pos_y)
    n = len(pos_x)

    par_dict = dict()
    # 先为每个类构建array(id)
    for _id in range(n):  # 由于在hypergraph构建的时候，增加了边界点，使每条边都有tail和head, 所以此处需要限制长度
        dict_append(par_dict, par_list[_id], _id)
    # 为每一个类构建[n, array(id), array(x), array(y)]
    for k, id_list in par_dict.items():
        id_list = np.array(id_list)
        x = pos_x[id_list]
        y = pos_y[id_list]
        nk = len(id_list)
        par_dict[k] = [nk, id_list, x, y]
    return par_dict


def eval_par(par_dict: dict):
    """
    越小越好
    """
    # TODO 改成 tensor 版本，加速
    # TODO 可能不够平稳
    # 先将partition转换成numpy.array
    # val = 1/N sum_k sum_i [(xi-x_bar)^2 + (yi-y_bar)^2]
    # N = 0
    val = 0
    val_list = []
    for k, (n, _id, x, y) in par_dict.items():
        # N += n
        val_tmp = np.sqrt(np.var(x) + np.var(y))
        # val += val_tmp * n
        val += val_tmp
        val_list.append(val_tmp)
    val /= len(par_dict)
    # val /= N
    return val, val_list


def eval_par_HPWL(par_list: list, hg):
    """
    越大越好
    """
    pos_x, pos_y = hg.pl
    hpwl_noncut = dict()  # 不被切割的边的hpwl
    num_nodes = len(pos_x)
    hpwl_cut = []  # 被切割边的hpwl
    for tail_list, head_list in hg.e2n:
        node_list = tail_list + head_list
        # HPWLe = max xi - min xi + max yi - min yi
        # 寻找 max xi, min xi, max yi, min yi
        xmax = ymax = 0
        xmin = ymin = 1e10
        xmax_id = xmin_id = ymax_id = ymin_id = 0
        found = True  # 标记是否在实际点中找到了最值位置
        is_cut = False
        real_node_list = [n for n in node_list if n < num_nodes]
        if len(real_node_list) < len(node_list):  # 这条超边连有添加的虚拟点，不考虑
            continue
        k = par_list[real_node_list[0]]
        for n in real_node_list:
            if par_list[n] != k:
                is_cut = True
            if pos_x[n] > xmax:
                xmax = pos_x[n]
                xmax_id = n
            if pos_x[n] < xmin:
                xmin = pos_x[n]
                xmin_id = n
            if pos_y[n] > ymax:
                ymax = pos_y[n]
                ymax_id = n
            if pos_y[n] < ymin:
                ymin = pos_y[n]
                ymin_id = n
        if not is_cut:  # 不被切割的边的 hpwl
            if k not in hpwl_noncut:
                hpwl_noncut[k] = []
            hpwl_noncut[k].append(xmax - xmin + ymax - ymin)
        else:  # 被切割边的 hpwl
            hpwl_cut.append(xmax - xmin + ymax - ymin)
    hpwl_k_noncut = [np.average(hpwl_k) for hpwl_k in hpwl_noncut.values()]
    total_hpwl_noncut = np.average(hpwl_k_noncut)
    totol_hpwl_cut = np.average(hpwl_cut)
    # return total_hpwl_noncut, hpwl_k_noncut
    return totol_hpwl_cut, hpwl_cut

def eval_par_weight(par_list,hg):
    """
    越小越好
    """
    pos_x, pos_y = hg.pl
    weight_noncut = dict()  # 不被切割的边的hpwl
    num_nodes = len(pos_x)
    weight_cut = []  # 被切割边的hpwl
    for eid,( tail_list, head_list )in enumerate(hg.e2n):
        node_list = tail_list + head_list
        is_cut = False
        real_node_list = [n for n in node_list if n < num_nodes]
        if len(real_node_list) < len(node_list):  # 这条超边连有添加的虚拟点，不考虑
            continue
        k = par_list[real_node_list[0]]
        for n in real_node_list:
            if par_list[n] != k:
                is_cut = True
        if not is_cut:  # 不被切割的边的 hpwl
            if k not in weight_noncut:
                weight_noncut[k] = []
            weight_noncut[k].append(hg.edge_weight[eid])
        else:  # 被切割边的 hpwl
            weight_cut.append(hg.edge_weight[eid])
    weight_k_noncut = [np.average(hpwl_k) for hpwl_k in weight_noncut.values()]
    total_hpwl_noncut = np.average(weight_k_noncut)
    totol_weight_cut = np.average(weight_cut)
    # return total_hpwl_noncut, hpwl_k_noncut
    return totol_weight_cut, weight_cut

def num_par(par: dict):
    """
    看各个切分的数量是否均衡
    """
    num = [n for n, _, _, _ in par.values()]
    var = np.var(num)
    # print(var)
    print(num)


def test_HPWL():
    from hypergraph import DiHypergraph

    hg = DiHypergraph()

    pos_x = [0, -2, 15, 10]
    pos_y = [4, -1, 2, 0]
    hg.e2n = [[[0, 1], [2, 3]]]
    hg.pl = [pos_x, pos_y]
    par_list = [0, 0, 0, 0]
    hpwl, _ = eval_par_HPWL(par_list, hg)
    assert hpwl == 22, "Error 1"

    pos_x = [0, 7, 5, 4, 12, 16]
    pos_y = [10, 14, 9, 2, 12, 5]
    hg.e2n = [[[0], [1, 2]], [[0], [2, 3]], [[2], [4, 5]], [[3], [5]]]
    hg.pl = [pos_x, pos_y]
    par_list = [0, 0, 0, 1, 0, 0]
    hpwl, _ = eval_par_HPWL(par_list, hg)
    assert hpwl == 30, "Error 2"


if __name__ == "__main__":
    # test_HPWL()
    pass
