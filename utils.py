import glob
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params


def check_pth(pth):
    if not os.path.exists(pth):
        os.system(f"mkdir ")


def del_ext_name(name: str):
    name = os.path.basename(name)
    name = name.split(".")[0]
    return name


def add_connected_subgraph(hg: nx.Graph, node_list, w=1):
    edge_list = [(s, t, w) for s in node_list for t in node_list if s < t]
    hg.add_weighted_edges_from(edge_list)
    return hg


def sort_by(arr1, arr2, reverse=False):
    """
    依据arr1对arr2排序
    reverse=True表示升序，反之降序
    """
    arr1, arr2 = np.array(arr1), np.array(arr2)
    idx = np.argsort(arr1)
    if reverse:
        idx = idx[::-1]
    return arr1[idx], arr2[idx]


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


def load_pl(pl_file):
    pos_x, pos_y = [], []
    with open(pl_file, encoding="utf-8") as f:
        # 跳过前两行
        f.readline()
        f.readline()
        for node in f:
            node = node.split()
            if len(node) >= 3:
                node_x, node_y = node[1:3]
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


def plot_pl_with_par(par: dict, vis_file):
    pos_x, pos_y, v_part = [], [], []
    for k, (n, kid, kx, ky) in par.items():
        pos_x.extend(kx)
        pos_y.extend(ky)
        v_part.extend([100 * k] * n)
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
    plt.close(fig)


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


def analysis_stats(res: str):
    # res = res.split("\n")[-3:-1]
    # run_time = float(res[0].split(":")[-1].replace("sec", ""))
    # io_time = float(res[1].split(":")[-1].replace("sec", ""))
    res = res[res.find("Partition time") :]
    res = res[: res.find("\n")]
    _, run_time = res.split("=")
    run_time = float(run_time.replace("s", ""))
    return run_time


def draw_density(hg):
    """
    绘制超图加边前后每个点权重比较
    """
    node_list = np.array(range(hg.num_node))
    weight = np.zeros(shape=(hg.num_node, 2))  # 第一位是算上虚拟边的总权重，第二位不算
    for i, (tail_list, head_list) in enumerate(hg.n2e):
        ori_tw = np.sum([hg.edge_width[e] for e in tail_list])
        ori_hw = np.sum([hg.edge_width[e] for e in head_list])
        ori_total = ori_tw + ori_hw
        weight[i][0] = weight[i][1] = ori_total
    for w, t, h in hg.vir_edge:
        weight[t][0] += w
        weight[h][0] += w
    weight[::-1].sort(axis=0)  # 降序

    total_weight = []
    ori_total_weight = []
    for total, ori_total in weight:
        total_weight.append(total)
        ori_total_weight.append(ori_total)

    pic_file = hg.hg_file + ".density.png"
    fig, ax = plt.subplots(dpi=300)
    ax.plot(node_list, total_weight, c="r", linewidth=0.5)
    ax.scatter(node_list, ori_total_weight, s=1)
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("weight")

    ax.text(
        4 / 7 * hg.num_node,
        1 / 2 * np.max(total_weight),
        r"$edge_{vir}-edge_{hg}$=" + str(len(hg.vir_edge)),
    )
    fig.savefig(pic_file, dpi=300)
    plt.close(fig)


def draw_node_density(hg):
    weight = np.zeros(shape=(hg.num_node,))  # 第一位是算上虚拟边的总权重，第二位不算
    for i, (tail_list, head_list) in enumerate(hg.n2e):
        ori_tw = np.sum([hg.edge_width[e] for e in tail_list])
        ori_hw = np.sum([hg.edge_width[e] for e in head_list])
        ori_total = ori_tw + ori_hw
        weight[i] = ori_total
    idx = np.argmax(weight)
    w = weight[idx]
    weight_ori = [hg.edge_width[e] for e in hg.n2e[idx][0]] + [hg.edge_width[e] for e in hg.n2e[idx][1]]
    weight_vir = []
    for w, t, h in hg.vir_edge:
        if t == idx or h == idx:
            weight_vir.append(w)

    weight_ori.sort(reverse=True)
    weight_vir.sort(reverse=True)
    pic_file = hg.hg_file + ".node_density.png"
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(range(len(weight_vir)), weight_vir, s=1, c="b")
    ax.scatter(range(len(weight_vir), len(weight_vir) + len(weight_ori)), weight_ori, s=1, c="r")
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("weight")
    ax.set_xlabel("edge")
    fig.savefig(pic_file, dpi=300)
    plt.close(fig)
