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
from utils import analysis_stats, load_par
from manager import generate_par, eval_par

k = 500


def dataflow_improve_random(hg: DiHypergraph, res_path, eta=[1.0]):
    print(f"dataflow_improve {hg.hg_src_file}")
    vir_edge_file = os.path.join(res_path, "vir_edge.bin")
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
    hg_list = []
    for etai in eta:
        vir_file = os.path.join(
            res_path, os.path.basename(hg.hg_src_file.replace(".hg", f".{int(100*etai):02d}.vir"))
        )
        hgi = copy.deepcopy(hg)
        if not os.path.exists(vir_file):
            N = int(len(vir_edge) * etai)
            vir_edge_i = random.sample(vir_edge, k=N)
            vir_edge_i.sort()
            hgi.add_vir_edge(vir_edge_i)
            hgi.write_dire(vir_file)
            hgi.write(vir_file.replace(".dire", ""))
        else:
            hgi.read_from_file(vir_file)
        hg_list.append(hgi)
    return hg_list


def par_one_file(hg: DiHypergraph, res_path, eta):
    hg_file = hg.hg_file
    print(hg_file, eta)
    res_name = os.path.basename(hg_file).replace(".vir", "")

    par_file_ori = hg_file + f".part.{k}"
    run_time_list, io_time_list, val_list = [], [], []
    for i in range(10):
        print(f"{i} shmetis {hg_file} {k} 2")
        status, res = subprocess.getstatusoutput(f"shmetis {hg_file} {k} 2")
        if status == 0:
            par_file = os.path.join(res_path, res_name + f".{i}")
            subprocess.getstatusoutput(f"mv {par_file_ori} {par_file}")
            res_file = par_file + ".res"
            with open(res_file, "w", encoding="utf8") as f:
                f.write(res)

            run_time, io_time = analysis_stats(res)
            par = load_par(par_file)
            par_dict = generate_par(par, hg.pl)
            val, _ = eval_par(par_dict)
            run_time_list.append(run_time)
            io_time_list.append(io_time)
            val_list.append(val)

            stat_file = par_file + ".stat"
            with open(stat_file, "w", encoding="utf8") as f:
                f.write(f"{eta:.4f} {run_time:.4f} {io_time:.4f} {val:.4f}\n")
        else:
            print(f"{i}: {eta}\n{res}")
    stat_key = res_name
    val = np.average(val_list)
    run_time = np.average(run_time_list)
    io_time = np.average(io_time_list)
    return stat_key, {
        "eta": eta,
        "value": val,
        "run_time": run_time,
        "io_time": io_time,
    }


def plot_num_res(val_base, stat_dict, res_pic):
    eval_list = [(v["eta"], v["value"]) for v in stat_dict.values()]
    eval_list.sort()
    eta = [etai for etai, _ in eval_list]
    val = [(vali - val_base) / val_base for _, vali in eval_list]

    fig, ax = plt.subplots()
    ax.set_xlabel("eta")
    ax.set_ylabel(
        r"$\frac{value_{vir}-value_{hg}}{value_{hg}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax.scatter(eta, val)
    fig.savefig(res_pic, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    random.seed(3407)
    conclude_file = "res/ispd2005/adaptec1/random_add/conclude.json"
    res_path = "res/ispd2005/adaptec1/random_add"
    res_pic = "res/ispd2005/adaptec1/random_add/conclude.png"
    subprocess.getstatusoutput(f"mkdir -p {res_path}")
    if not os.path.exists(conclude_file):
        hg_file = "res/ispd2005/adaptec1/adaptec1.hg.dire"
        pl_file = "res/ispd2005/adaptec1/adaptec1.gp.pl"
        hg = DiHypergraph()
        hg.read_from_file(hg_file)
        hg.read_pl(pl_file)
        # eta = np.linspace(0, 1, 11)
        eta = [0.6, 0.7, 0.8, 0.9, 1.0]
        print("dataflow improve")
        hg_list = dataflow_improve_random(hg, res_path, eta)

        print("start partition")
        p = Pool(5)
        task_list = []
        for hgi, etai in zip(hg_list, eta):
            task_list.append(p.apply_async(par_one_file, (hgi, res_path, etai)))
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
    else:
        with open(conclude_file) as f:
            stat_dict = jstyleson.load(f)

    print("plot conclude")
    base = {2: 2708.3729399999997, 50: 1026.2599500000001, 500: 637.6539, 1024: 605.0860700000001}
    val_base = base[k]
    plot_num_res(val_base, stat_dict, res_pic)
