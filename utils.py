import glob
import networkx as nx
import matplotlib.pyplot as plt
import os


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
    hg, is_weight_vec, vec_weight_list = generate_hg(gfile)
    with open(pfile, encoding="utf-8") as f:
        v_part = [int(p) for p in f]
    # if is_weight_vec:
    nx.draw_networkx(
        hg,
        # pos=nx.spring_layout(hg, iterations=150),
        pos=nx.shell_layout(hg),
        with_labels=True,
        edge_color="gray",
        node_color=v_part,
        cmap=plt.cm.rainbow,
        node_size=[250 + w * 80 for w in vec_weight_list] if is_weight_vec else 300,
    )
    plt.savefig(f"{dst}/{os.path.basename(pfile)}.png")
    plt.clf()
    hg.clear()
