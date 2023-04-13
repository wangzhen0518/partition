import numpy as np
import jstyleson
import matplotlib.pyplot as plt
import os

from utils import load_position, load_par, dict_append, del_ext_name, generate_benchmark_dict

"""
partition的数据结构为
{
    ...,
    k: [n, array(id), array(xi), array(yi)],
    ...
}

n: 第k个类中点的数量
"""


def generate_par(par, pl):
    pos_x, pos_y = pl
    pos_x, pos_y = np.array(pos_x), np.array(pos_y)
    n = len(pos_x)

    par_dict = dict()
    # 先为每个类构建array(id)
    for _id in range(n):  # 由于在hypergraph构建的时候，增加了边界点，使每条边都有tail和head, 所以此处需要限制长度
        dict_append(par_dict, par[_id], _id)
    # 为每一个类构建[n, array(id), array(x), array(y)]
    for k, id_list in par_dict.items():
        id_list = np.array(id_list)
        x = pos_x[id_list]
        y = pos_y[id_list]
        nk = len(id_list)
        par_dict[k] = [nk, id_list, x, y]
    return par_dict


def eval_par(par: dict):
    # TODO 改成 tensor 版本，加速
    # TODO 可能不够平稳
    # 先将partition转换成numpy.array
    # val = 1/N sum_k sum_i [(xi-x_bar)^2 + (yi-y_bar)^2]
    # N = 0
    val = 0
    val_list = []
    for k, (n, _id, x, y) in par.items():
        # N += n
        val_tmp = np.sqrt(np.var(x) + np.var(y))
        # val += val_tmp * n
        val += val_tmp
        val_list.append(val_tmp)
    val /= len(par)
    # val /= N
    return val, val_list


def num_par(par: dict):
    """
    看各个切分的数量是否均衡
    """
    num = [n for n, _, _, _ in par.values()]
    var = np.var(num)
    # print(var)
    print(num)


def sort_keys(key_lst, type):
    idx_lst = []
    k_lst = []
    if type == "k":
        tmp = []
        for k in key_lst:
            idx = int(k.split(".")[1])
            tmp.append((idx, k))
        tmp.sort()
        for idx, k in tmp:
            idx_lst.append(idx)
            k_lst.append(k)
    elif type == "g":
        for k in key_lst:
            idx = k.split(".")[0]
            idx_lst.append(idx)
            k_lst.append(k)
    return idx_lst, k_lst


def compare(vir_conclude, hg_conclude, type):
    cmp_f = lambda x, y: (x - y) / y
    with open(vir_conclude, "r", encoding="utf-8") as f:
        vir_conclude = jstyleson.load(f)
    with open(hg_conclude, "r", encoding="utf-8") as f:
        hg_conclude = jstyleson.load(f)
    idx_lst, k_lst = sort_keys(vir_conclude.keys(), type)
    cmp_lst = []
    for t in k_lst:
        cmp_lst.append(cmp_f(vir_conclude[t]["value"], hg_conclude[t]["value"]))
    return idx_lst, k_lst, cmp_lst


def plot_cmp(cmp_lst, idx_lst, type):
    fig, ax = plt.subplots()
    x = list(range(len(idx_lst)))
    ax.scatter(x, cmp_lst)
    # ax.scatter(x, cmp_ntup_lst)
    ax.axhline(y=0, c="r", ls="-")
    ax.set_ylabel(
        r"$\frac{value_{vir}-value_{hg}}{value_{hg}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    idx_r = idx_lst[0]
    x_l = x_r = x[0]  # x_l 是区间左端点, x_r 是区间右端点
    for i, (xi, idx) in enumerate(zip(x, idx_lst)):
        if idx != idx_r:
            x_r = x[i - 1]
            ax.axvline(x=(x_r + xi) / 2, c="orange", ls="--")
            ax.annotate(
                text=idx_r,
                xy=((x_l + x_r) / 2, 0.1),
                xytext=((x_l + x_r) / 2 - len(str(idx_r)) / (x_r - x_l) / 2, 0.1),
                c="darkred",
                weight="bold",
            )
            print(idx_r)
            idx_r = idx
            x_l = xi
    x_r = x[-1]
    ax.annotate(
        text=idx_r,
        xy=((x_l + x_r) / 2, 0.1),
        xytext=((x_l + x_r) / 2 - len(str(idx_r)) / (x_r - x_l) / 2, 0.1),
        c="darkred",
        weight="bold",
    )
    print(idx_r)
    fig.savefig(f"res/ispd2005/conclude.shmetis.{type}.png", dpi=300)
    plt.close(fig)
    # print(idx_lst)


def plot_improve(design_lst):
    def get_num_edges(file):
        with open(file, encoding="utf8") as f:
            num_edges, _, _ = f.readline().split()
            num_edges = int(num_edges)
            return num_edges

    cmp_f = lambda x, y: x - y
    cmp_lst = []
    for design in design_lst:
        hg_file = os.path.join("res", "ispd2005", design, design + ".hg")
        vir_file = os.path.join("res", "ispd2005", design, design + ".vir")
        hg_num_edges = get_num_edges(hg_file)
        vir_num_edges = get_num_edges(vir_file)
        cmp_lst.append(cmp_f(vir_num_edges, hg_num_edges))
    fig, ax = plt.subplots()
    ax.scatter(design_lst, cmp_lst)
    ax.axhline(y=0, c="r", ls="-")
    ax.set_ylabel(
        # r"$\frac{edge_{vir}-edge_{hg}}{edge_{hg}}$",
        r"$edge_{vir}-edge_{hg}$",
        fontdict={"size": 12},
        rotation=0,
        loc="top",
        labelpad=-80,
    )
    fig.savefig("res/ispd2005/conclude.shmetis.num_edge.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    type = "g"  # 'k' or 'g', k 表示按照切分数量优先, g 表示按照图有限
    vir_gp_conclude = "res/ispd2005/conclude.vir.shmetis.gp.json"
    hg_gp_conclude = "res/ispd2005/conclude.hg.shmetis.gp.json"
    idx_lst, k_lst, cmp_gp_lst = compare(vir_gp_conclude, hg_gp_conclude, type)
    plot_cmp(cmp_gp_lst, idx_lst, type)

    benchmark = "ispd2005"
    config_file = os.path.join("par_config", benchmark, "config.json")
    with open(config_file, encoding="utf-8") as f:
        config = jstyleson.load(f)
    plot_improve(config["design"])
