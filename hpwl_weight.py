import os
import jstyleson
import subprocess
import numpy as np
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle as pk

from hypergraph import DiHypergraph, DiGraph
from manager import generate_par, eval_par, eval_par_HPWL
from utils import plot_pl_with_par, del_ext_name, analysis_stats, load_par
from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB


def generate_hg(design, new_hg=False):
    hg = DiHypergraph()
    pl_file = os.path.join("res", "ispd2005", design, f"{design}.gp.pl")
    hg.read_pl(pl_file)
    hg_file = os.path.join("res", "ispd2005", design, f"{design}.hg.dire")
    if os.path.exists(hg_file) and not new_hg:
        hg.read_from_file(hg_file)
    else:
        pl_config = os.path.join("pl_config", "ispd2005", design + ".json")
        hg.build_from_config(pl_config, hg_file)
    return hg


def generate_g(design, new_g=False):
    g = DiGraph()
    pl_file = os.path.join("res", "ispd2005", design, f"{design}.gp.pl")
    g.read_pl(pl_file)
    hg = generate_hg(design)
    g_file = os.path.join("res", "ispd2005", design, f"{design}.100.hg.dire")
    if os.path.exists(g_file) and not new_g:
        g.read_from_file(g_file)
        g.hg = hg
        g.pl = copy.deepcopy(hg.pl)
    else:
        g.read_from_hg(hg, 0.8)
        g.write_dire(g_file)
        g.write(g_file.replace(".dire", ""))
    return g


def hpwl_weight(g: DiGraph):
    hw = []
    pos_x, pos_y = g.pl
    n = len(pos_x)
    for n1, (tail_list, head_list) in enumerate(g.graph[:n]):
        for n2, w in tail_list:
            if n2 < n:
                hpwl = abs(pos_x[n1] - pos_x[n2]) + abs(pos_y[n1] - pos_y[n2])
                hw.append([hpwl, w])
    hw.sort()
    return hw


if __name__ == "__main__":
    benchmark = "ispd2005"
    b_pth = os.path.join("res", benchmark)
    os.system(f"mkdir -p {b_pth}")
    config_file = os.path.join("par_config", benchmark, "config.json")
    with open(config_file, encoding="utf-8") as f:
        config = jstyleson.load(f)
    g = generate_g("bigblue2")
    hw = hpwl_weight(g)
    hpwl, weight = [], []
    for h, w in hw:
        hpwl.append(h)
        weight.append(w)
    fig, ax = plt.subplots()
    ax.scatter(hpwl, weight, s=1)
    ax.set_xlabel("hpwl")
    ax.set_ylabel("weight")
    fig.savefig("res/ispd2005/conclude/hpwl_weight.png", dpi=300)
    plt.close(fig)
