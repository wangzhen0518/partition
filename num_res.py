import numpy as np
from multiprocessing import Pool
import subprocess
import os
import jstyleson
import matplotlib.pyplot as plt
import matplotlib
import random
import copy
import pickle as pk

from hypergraph import DiHypergraph
from utils import analysis_stats, load_par, load_pl, plot_pl_with_par, draw_density, draw_node_density
from manager import generate_par, eval_par, eval_par_HPWL

matplotlib.use("Agg")

k = 500


def dataflow_improve_eta(hg: DiHypergraph, res_path, eta=[1.0], density_func=None):
    print(f"dataflow_improve {hg.hg_src_file}")
    # get vir_edge
    vir_edge_file = os.path.join(res_path, "vir_edge.bin")
    if os.path.exists(vir_edge_file):
        with open(vir_edge_file, "rb") as f:
            vir_edge = pk.load(f)
    else:
        vir_edge = hg.cal_dataflow2(k=3)
        with open(vir_edge_file, "wb") as f:
            pk.dump(vir_edge, f)
    # generate hg_list
    hg_list = []
    for etai in eta:
        # generate hgi
        vir_file = os.path.join(
            res_path, os.path.basename(hg.hg_src_file.replace(".hg", f".{int(100*etai):02d}.vir"))
        )
        hgi = copy.deepcopy(hg)
        if not os.path.exists(vir_file):
            N = int(len(vir_edge) * etai)
            vir_edge_i = random.sample(vir_edge, k=N)
            # vir_edge_i.sort()
            hgi.add_vir_edge(vir_edge_i)
            hgi.write_dire(vir_file)
            hgi.write(vir_file.replace(".dire", ""))
        else:
            hgi.read_from_file(vir_file)
        hg_list.append(hgi)
        if density_func:
            density_func(hgi)
    return hg_list


def par_one_file(hg: DiHypergraph, res_path, eta, new_par=False):
    hg_file = hg.hg_file
    print(hg_file, eta)
    res_name = os.path.basename(hg_file).replace(".vir", "")

    par_file_ori = hg_file + f".part.{k}"
    run_time_list, io_time_list, val_list, hpwl_list = [], [], [], []
    for i in range(10):
        par_file = os.path.join(res_path, res_name + f".{i}")
        res_file = par_file + ".res"
        if os.path.exists(par_file) and os.path.exists(res_file) and not new_par:
            with open(res_file, "r", encoding="utf8") as f:
                res = f.read()
        else:
            print(f"{i} shmetis {hg_file} {k} 2")
            status, res = subprocess.getstatusoutput(f"shmetis {hg_file} {k} 2")
            if status == 0:
                subprocess.getstatusoutput(f"mv {par_file_ori} {par_file}")
                with open(res_file, "w", encoding="utf8") as f:
                    f.write(res)
            else:
                print(f"{i}: {eta}\n{res}")
                continue
        run_time, io_time = analysis_stats(res)
        par_list = load_par(par_file)
        par_dict = generate_par(par_list, hg.pl)
        val, _ = eval_par(par_dict)
        hpwl, _ = eval_par_HPWL(par_list, hg)
        run_time_list.append(run_time)
        io_time_list.append(io_time)
        val_list.append(val)
        hpwl_list.append(hpwl)

        stat_file = par_file + ".stat"
        with open(stat_file, "w", encoding="utf8") as f:
            f.write(f"{eta:.4f} {run_time:.4f} {io_time:.4f} {val:.4f}\n")

    stat_key = res_name
    best_idx = int(np.argmin(val_list))
    val = np.average(val_list)
    hpwl = np.average(hpwl_list)
    run_time = np.average(run_time_list)
    io_time = np.average(io_time_list)
    return stat_key, {
        "eta": eta,
        "value": val,
        "hpwl": hpwl,
        "run_time": run_time,
        "io_time": io_time,
        "best": best_idx,
    }


def run_and_conclude(
    hg_file, pl_file, eta, res_path, conclude_file, new_conclude=False, n=5, density_func=None
):
    if os.path.exists(conclude_file) and not new_conclude:
        with open(conclude_file) as f:
            stat_dict = jstyleson.load(f)
    else:
        hg = DiHypergraph()
        hg.read_from_file(hg_file)
        hg.read_pl(pl_file)
        print("dataflow improve")
        hg_list = dataflow_improve_eta(hg, res_path, eta, density_func)
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


def visualize_results(design, pl_file, res_path, stat_dict):
    for key, stat_info in stat_dict.items():
        etai = stat_info["eta"]
        idx = int(stat_info["best"])
        pl = load_pl(pl_file)
        par_file = os.path.join(res_path, f"{key}.{idx}")
        par = load_par(par_file)
        par_dict = generate_par(par, pl)
        pic = os.path.join(res_path, f"{key}.{idx}.png")
        plot_pl_with_par(par_dict, pic)


def plot_conclude(stat_dict, res_pic, val_base=None, hpwl_base=None):
    eval_list = [(v["eta"], v["value"], v["hpwl"]) for v in stat_dict.values()]
    eval_list.sort()
    if val_base is None:
        val_base = eval_list[0][1]
    eta, val, hpwl = [], [], []
    for etai, vali, hpwli in eval_list:
        eta.append(etai)
        val.append((vali - val_base) / val_base if val_base else vali)
        hpwl.append((hpwli - hpwl_base) / hpwl_base if hpwl_base else hpwli)

    fig, ax = plt.subplots()
    # plot value
    ax.set_xlabel("eta")
    ax.set_ylabel(
        r"$\frac{value_{vir}-value_{hg}}{value_{hg}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    l1 = ax.scatter(eta, val, c="b")
    # plot hpwl
    # ax2 = ax.twinx()
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel(
        r"$-\frac{hpwl_{vir}-hpwl_{hg}}{hpwl_{hg}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    l2 = ax2.scatter(eta, hpwl, c="r")

    # ax.legend((l1, l2), ("value", "hpwl"), loc=0)
    fig.savefig(res_pic, dpi=300)
    fig2.savefig(res_pic.replace(".png", ".hpwl.png"), dpi=300)
    plt.close(fig)
    plt.close(fig2)


def draw_res():
    random.seed(3407)
    design = "adaptec2"
    conclude_file = f"res/ispd2005/{design}/num/conclude.json"
    hg_file = f"res/ispd2005/{design}/{design}.hg.dire"
    pl_file = f"res/ispd2005/{design}/{design}.gp.pl"
    res_path = f"res/ispd2005/{design}/num"
    subprocess.getstatusoutput(f"mkdir -p {res_path}")

    print("partition")
    n = 2
    eta = np.linspace(0.8, 1, 5)
    # eta = [0.8]
    stat_dict = run_and_conclude(hg_file, pl_file, eta, res_path, conclude_file, True, n, draw_density)

    print("draw partition")
    visualize_results(design, pl_file, res_path, stat_dict)

    print("plot conclude")
    # adaptec1
    # base = {2: 2708.3729399999997, 50: 1026.2599500000001, 500: 637.6539, 1024: 605.0860700000001}
    # base_hpwl = {2: 34718549.4, 50: 1053319.418, 500: 81672.04520000001, 1024: 31404.43056640625}

    # adaptec2
    base = {2: 2708.3729399999997, 50: 1026.2599500000001, 500: 793.9420945050233, 1024: 605.0860700000001}
    base_hpwl = {2: 34718549.4, 50: 1053319.418, 500: 90231.3756, 1024: 31404.43056640625}

    # adaptec3
    # base = {2: 2708.3729399999997, 50: 1026.2599500000001, 500: 1180.4426862211462, 1024: 605.0860700000001}
    # base_hpwl = {2: 34718549.4, 50: 1053319.418, 500: 175888.09480000002, 1024: 31404.43056640625}

    # adaptec4
    # base = {2: 2708.3729399999997, 50: 1026.2599500000001, 500: 1182.0290735881504, 1024: 605.0860700000001}
    # base_hpwl = {2: 34718549.4, 50: 1053319.418, 500: 128529.03799999999, 1024: 31404.43056640625}

    val_base = base[k]
    hpwl_base = base_hpwl[k]
    res_pic = f"res/ispd2005/{design}/num/conclude.png"
    plot_conclude(stat_dict, res_pic, val_base, hpwl_base)


def draw_node_density_instance():
    random.seed(3407)
    design = "adaptec2"
    conclude_file = f"res/ispd2005/{design}/num/conclude.json"
    hg_file = f"res/ispd2005/{design}/{design}.hg.dire"
    pl_file = f"res/ispd2005/{design}/{design}.gp.pl"
    res_path = f"res/ispd2005/{design}/num"
    subprocess.getstatusoutput(f"mkdir -p {res_path}")

    n = 2
    eta = np.linspace(0.8, 1, 5)

    hg = DiHypergraph()
    hg.read_from_file(hg_file)
    hg.read_pl(pl_file)
    print("dataflow improve")
    dataflow_improve_eta(hg, res_path, eta, draw_node_density)


if __name__ == "__main__":
    draw_node_density_instance()
