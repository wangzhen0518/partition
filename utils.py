import glob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params


def del_ext_name(name: str):
    name = os.path.basename(name)
    name = name.split(".")[0]
    return name


def add_connected_subgraph(hg: nx.Graph, node_list, w=1):
    edge_list = [(s, t, w) for s in node_list for t in node_list if s < t]
    hg.add_weighted_edges_from(edge_list)
    return hg


def generate_hg(gfile):
    """
    根据 gfile 生成 hypergraph
    """
    hg = nx.Graph()
    edge_vec_list = []
    vec_weight_list = []
    is_weight_vec = False
    is_weight_edge = False
    with open(gfile, encoding="utf-8") as f:
        ginfo = [int(n) for n in f.readline().split()]
        nedge, nvec = ginfo[:2]  # 第一行, nedge, nvec
        if len(ginfo) == 3:
            if ginfo[-1] == 1 or ginfo[-1] == 11:
                is_weight_edge = True
            if ginfo[-1] == 10 or ginfo[-1] == 11:
                is_weight_vec = True
        for e in range(nedge):
            edge_vec_list.append([int(vec) - 1 for vec in f.readline().split()])  # vertex 的序号从 1 开始，此处还原为 0
        if is_weight_vec:
            for v in range(nvec):
                vec_weight_list.append(int(f.readline()))
    hg.add_nodes_from(range(nvec))
    if is_weight_edge:
        for node_list in edge_vec_list:
            add_connected_subgraph(hg, node_list[1:], node_list[1])
    else:
        for node_list in edge_vec_list:
            add_connected_subgraph(hg, node_list)

    return hg, is_weight_vec, vec_weight_list


def load_position(pl_file):
    pos_x, pos_y = [], []
    with open(pl_file, encoding="utf-8") as f:
        # 跳过前两行
        f.readline()
        f.readline()
        for node in f:
            node_x, node_y = node.split()[1:3]
            node_x, node_y = int(node_x), int(node_y)
            pos_x.append(node_x)
            pos_y.append(node_y)
    return pos_x, pos_y


def load_par(par_file):
    with open(par_file, encoding="utf-8") as f:
        v_part = [int(p) for p in f]
    return v_part


def visualize_graph(gfile, pfile, dst):
    """
    分两步
    1. 根据 gfile 生成 hypergraph
    2. 根据 .part 对图上点进行着色

    gfile: graph files
    pfile: partition result files
    k: # of partition
    """
    print(f"{gfile}:\t{pfile}")
    hg, is_weight_vec, vec_weight_list = generate_hg(gfile)
    v_part = load_par(pfile)
    # if is_weight_vec:
    # nx.draw_networkx(
    #     hg,
    #     pos=nx.spring_layout(hg, iterations=50, seed=3407, weight=None),
    #     # pos=nx.shell_layout(hg),
    #     # with_labels=True,
    #     edge_color="grey",
    #     node_color=v_part,
    #     cmap=plt.cm.rainbow,
    #     alpha=0,
    #     # node_size=[250 + w * 80 for w in vec_weight_list] if is_weight_vec else 300,
    #     node_size=10,
    # )
    _pos = nx.spring_layout(hg, iterations=10, seed=3407, weight=None)
    # _pos = nx.shell_layout(hg)
    plt.figure(dpi=300)
    nx.draw_networkx_nodes(
        hg,
        pos=_pos,
        # pos=nx.shell_layout(hg),
        # with_labels=True,
        node_color=v_part,
        cmap=plt.cm.rainbow,
        node_size=1,
    )
    plt.savefig(f"{dst}/{os.path.basename(pfile).replace('.hg','')}.png", dpi=300)
    plt.clf()
    hg.clear()


def plot_pl_with_par(pl_file, par_file, vis_file):
    pos_x, pos_y = load_position(pl_file)
    v_part = load_par(par_file)
    fig, ax = plt.subplots(dpi=1000)
    ax.scatter(
        pos_x,
        pos_y,
        s=0.5,
        c=v_part,
        edgecolors="none",
        cmap=plt.cm.jet,
    )
    fig.savefig(vis_file, dpi=1000)


def check_hg_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        f.readline()  # 跳过第一行
        for i, l in enumerate(f):
            v_list = [int(v) for v in l.split()]
            if len(v_list) <= 1:
                print(f"Error: line {i+2}")


def dict_append(d: dict, k, v):
    """
    d 的 value 是 list, 向该 list 中添加元素 v
    """
    if k not in d:
        d[k] = []
    d[k].append(v)


def generate_benchmark_dict(benchmark, method):
    """
    bench_dict={
        design:{
            'hg': hg_file,
            'pl': pl_file,
            'par': par_list,
            'stats': stats_list,
            'vis': vis_list | None
        }
    }
    """
    bench_dict = dict()
    hg_list = glob.glob(os.path.join("res", benchmark, "hypergraph", "*.hg"))
    pl_list = glob.glob(os.path.join("res", benchmark, "pl", "*.pl"))
    hg_list.sort()
    pl_list.sort()
    design_list = [del_ext_name(hg_file) for hg_file in hg_list]
    for design, hg_file, pl_file in zip(design_list, hg_list, pl_list):
        par_list = glob.glob(os.path.join("res", benchmark, method, "par", design + "*"))
        par_list.sort()
        bench_dict[design] = {"hg": hg_file, "pl": pl_file, "par": par_list}
    return bench_dict
