import numpy as np
from multiprocessing import Pool
import subprocess
import os
import jstyleson
import matplotlib.pyplot as plt
import random
import copy
import pickle as pk

from hypergraph import DiHypergraph
from utils import analysis_stats, load_par, plot_pl_with_par
from manager import generate_par, eval_par, eval_par_HPWL

k = 2


def generate_one_hg(hg, etai, vir_file, vir_edge):
    hgi = DiHypergraph()
    if not os.path.exists(vir_file):
        N = int(len(vir_edge) * etai)
        vir_edge_i = random.sample(vir_edge, k=N)
        hgi = copy.deepcopy(hg)
        hgi.add_vir_edge(vir_edge_i)
        hgi.write_dire(vir_file)
        hgi.write(vir_file.replace(".dire", ""))
    else:
        hgi.read_from_file(vir_file)
        hgi.pl = copy.deepcopy(hg.pl)
    return etai, hgi


def dataflow_improve_random(hg: DiHypergraph, res_path, eta=[1.0], n=8):
    print(f"dataflow_improve {hg.hg_src_file}")
    vir_edge_file = os.path.join(res_path, "vir_edge.random_add.bin")
    if os.path.exists(vir_edge_file):
        with open(vir_edge_file, "rb") as f:
            vir_edge = pk.load(f)
    else:
        # 随机生成 num_edge 条边
        vir_edge = []
        for i in range(hg.num_edge):
            s, t = random.sample(range(hg.num_node), 2)
            w = random.choice(range(5, 11))
            vir_edge.append((w, s, t))
        with open(vir_edge_file, "wb") as f:
            pk.dump(vir_edge, f)
    task_list = []
    p = Pool(n)
    for etai in eta:
        vir_file = os.path.join(
            res_path, os.path.basename(hg.hg_src_file.replace(".hg", f".{int(100*etai):02d}.vir"))
        )
        task_list.append(p.apply_async(generate_one_hg, args=(hg, etai, vir_file, vir_edge)))
    p.close()
    p.join()

    hg_list = []
    for t in task_list:
        hg_list.append(t.get())
    hg_list.sort()
    hg_list = [hgi for _, hgi in hg_list]
    return hg_list


def par_one_file(hg: DiHypergraph, res_path, eta, new_par=False):
    hg_file = hg.hg_file
    # print(hg_file, eta)
    res_name = os.path.basename(hg_file).replace(".vir", "")

    par_file_ori = f"{hg_file}.part{k}.epsilon0.03.seed-1.KaHyPar"
    par_file = os.path.join(res_path, res_name)
    res_file = par_file + ".res"
    if os.path.exists(par_file) and os.path.exists(res_file) and not new_par:
        print(f"{par_file}.res exists")
        with open(res_file, "r", encoding="utf8") as f:
            res = f.read()
    else:
        cmd = f"KaHyPar -h {hg_file} -k {k} -e 0.03 -o km1 -m direct -p ./kahypar_config/km1_kKaHyPar_sea20.ini -w 1"
        print(eta, cmd)
        status, res = subprocess.getstatusoutput(cmd)
        if status == 0:
            subprocess.getstatusoutput(f"mv {par_file_ori} {par_file}")
            with open(res_file, "w", encoding="utf8") as f:
                f.write(res)
        else:
            raise RuntimeError(f"{eta} Error: {res}")
    run_time = analysis_stats(res)
    par_list = load_par(par_file)
    par_dict = generate_par(par_list, hg.pl)
    val, val_list = eval_par(par_dict)
    hpwl, hpwl_list = eval_par_HPWL(par_list, hg)

    plot_pl_with_par(par_dict, par_file + ".pdf")

    stat_key = res_name
    return stat_key, {
        "value": val,
        "hpwl": hpwl,
        "ncut": len(hpwl_list),
        "run_time": run_time,
        "eta": eta,
    }


def get_baseline():
    base_file = os.path.join("res", "ispd2005", "conclude.hg.origin.json")
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = jstyleson.load(f)
    base_dict = baseline
    return base_dict


def plot_conclude(stat_dict, res_pic_name, val_base, hpwl_base, ncut_base):
    """
    统计切分结果，并可视化
    """
    eval_list = [(v["eta"], v["value"], v["hpwl"], v["ncut"]) for v in stat_dict.values()]
    eval_list.sort()
    eta, val, hpwl, ncut = [], [], [], []
    for etai, vali, hpwli, ncuti in eval_list:
        eta.append(etai)
        val.append((vali - val_base) / val_base if val_base else vali)
        hpwl.append(-(hpwli - hpwl_base) / hpwl_base if hpwl_base else hpwli)
        ncut.append((ncuti - ncut_base) / ncut_base if hpwl_base else hpwli)

    fig, ax = plt.subplots()
    # plot value
    ax.set_xlabel("eta")
    ax.set_ylabel(
        r"$\frac{value_{add}-value_{ori}}{value_{ori}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax.scatter(eta, val, c="b")
    fig.savefig(res_pic_name + ".value.pdf", dpi=300)
    plt.close(fig)

    # plot hpwl
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel(
        r"$-\frac{hpwl_{add}-hpwl_{ori}}{hpwl_{ori}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax2.scatter(eta, hpwl, c="r")
    fig2.savefig(res_pic_name + ".hpwl.pdf", dpi=300)
    plt.close(fig2)

    # plot ncut
    fig3, ax3 = plt.subplots()
    ax3.set_ylabel(
        r"$\frac{ncut_{add}-ncut_{ori}}{ncut_{ori}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax3.scatter(eta, ncut, c="c")
    fig3.savefig(res_pic_name + ".ncut.pdf", dpi=300)
    plt.close(fig3)


def run_and_conclude(hg_file, pl_file, eta, res_path, conclude_file, new_conclude=False, n=5):
    """
    生成并切分所有超图
    """
    if os.path.exists(conclude_file) and not new_conclude:
        with open(conclude_file) as f:
            stat_dict = jstyleson.load(f)
    else:
        hg = DiHypergraph()
        hg.read_from_file(hg_file)
        hg.read_pl(pl_file)
        print("dataflow improve")
        hg_list = dataflow_improve_random(hg, res_path, eta, n)
        print("start partition")
        p = Pool(n)
        task_list = []
        for hgi, etai in zip(hg_list, eta):
            task_list.append(p.apply_async(par_one_file, (hgi, res_path, etai, False)))
        p.close()
        p.join()
        print("start conclude")
        stat_dict = dict()
        for task in task_list:
            stat_key, stat_info = task.get()
            stat_dict[stat_key] = stat_info
        print("get conclude dictionary")
        with open(conclude_file, "w", encoding="utf8") as f:
            jstyleson.dump(stat_dict, f, sort_keys=True, indent=4)
    return stat_dict


def random_add(design):
    hg_file = f"res/ispd2005/{design}/{design}.hg.dire"
    pl_file = f"res/ispd2005/{design}/{design}.gp.pl"
    res_path = f"res/ispd2005/{design}/random_add"
    conclude_file = os.path.join(res_path, "conclude.json")
    subprocess.getstatusoutput(f"mkdir -p {res_path}")

    print("partition")
    n = 8
    eta = np.linspace(0, 1, 21)
    stat_dict = run_and_conclude(hg_file, pl_file, eta, res_path, conclude_file, False, n)

    print("plot conclude")
    base_dict = get_baseline()
    base_name = design + f".{k}"
    val_base = base_dict[base_name]["value"]
    hpwl_base = base_dict[base_name]["hpwl"]
    ncut_base = base_dict[base_name]["ncut"]
    res_pic_name = os.path.join(res_path,"conclude")
    plot_conclude(stat_dict, res_pic_name, val_base, hpwl_base, ncut_base)


if __name__ == "__main__":
    random.seed(3407)
    np.random.seed(3407)
    # random_add("adaptec1")
    # random_add("adaptec2")
    # random_add("adaptec3")
    # random_add("adaptec4")
    random_add("bigblue1")
    random_add("bigblue2")
    random_add("bigblue3")
    # random_add("bigblue4")
