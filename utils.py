import glob
import networkx as nx
import matplotlib.pyplot as plt
import os

import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params


def del_ext_name(name: str):
    name = os.path.basename(name)
    name = name.split(".")[0]
    return name


def generate_single_hg_file(src, dst):
    params = Params.Params()
    placedb = PlaceDB.PlaceDB()
    params.load(src)
    placedb(params)

    # 遍历 placedb.net2pin_map, placedb.net_weights
    edge_list = []
    del_edge_cnt = 0
    for e, w in zip(placedb.net2pin_map, placedb.net_weights):
        e = set([placedb.pin2node_map[p] + 1 for p in e])
        if len(e) == 1:
            del_edge_cnt += 1
            continue
        # edge_list.append([int(w), *e])
        edge_list.append([*e])
    with open(dst, "w", encoding="utf-8") as f:
        # f.write(f"{placedb.num_nets} {placedb.num_physical_nodes} 1\n")  # 1 为加权边
        f.write(f"{placedb.num_nets-del_edge_cnt} {placedb.num_physical_nodes}\n")
        for e in edge_list:
            f.write(" ".join([str(v) for v in e]) + "\n")
    return placedb


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
    with open(pfile, encoding="utf-8") as f:
        v_part = [int(p) for p in f]
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
