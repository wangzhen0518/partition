import subprocess
import os
import jstyleson
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import copy
import random

from multiprocessing import Pool

from utils import analysis_stats, load_par, load_pl, plot_pl_with_par, draw_density, draw_node_density
from manager import generate_par, eval_par, eval_par_HPWL
from hypergraph import DiHypergraph


def stable(design):
    """
    KaHyPar是否稳定
    """
    hg_file = f"res/ispd2005/{design}/{design}.hg"
    pl_file = f"res/ispd2005/{design}/{design}.gp.pl"
    res_json = f"res/ispd2005/{design}/stable.ln.json"
    res_fig = f"res/ispd2005/{design}/stable.ln.png"
    k = 500
    N = 10
    res_dict = dict()
    for i in range(N):
        cmd = f"KaHyPar -h {hg_file} -k {k} -e 0.03 -o km1 -m direct -p ./kahypar_config/km1_kKaHyPar_sea20.ini -w 1"
        print(i, cmd)
        stat, res = subprocess.getstatusoutput(cmd)
        target_file = hg_file + f".{k}.{i}"
        if stat == 0:
            os.system(f"mv {hg_file}.part{k}.epsilon0.03.seed-1.KaHyPar {target_file}")
            par_list = load_par(target_file)
            pl = load_pl(pl_file)
            par_dict = generate_par(par_list, pl)
            val, _ = eval_par(par_dict)
            key = os.path.basename(target_file)
            res_dict[key] = np.log(val)
            print(key, val)
        else:
            print(f"Error {res}")
    with open(res_json, "w", encoding="utf8") as f:
        jstyleson.dump(res_dict, f, sort_keys=True, indent=4)
    val_lst = np.array(list(res_dict.values()))
    val_aver = np.average(val_lst)
    var = np.var(val_lst)
    val_lst = (val_lst - val_aver) / val_aver
    plt.scatter(range(len(val_lst)), val_lst)
    plt.axhline(y=0, c="r", ls="-")
    plt.text(0, 0.05, f"$\ln(var)$={np.log(var):.2f}")
    plt.savefig(res_fig, dpi=300)
    plt.close()


k = 500


def num_value(design):
    """
    切分效果随加边数量变化
    """
    random.seed(3407)
    conclude_file = f"res/ispd2005/{design}/num/conclude.json"
    hg_file = f"res/ispd2005/{design}/{design}.hg.dire"
    pl_file = f"res/ispd2005/{design}/{design}.gp.pl"
    res_path = f"res/ispd2005/{design}/num"
    subprocess.getstatusoutput(f"mkdir -p {res_path}")

    print("partition")
    n = 8
    eta = np.linspace(0, 1, 21)
    # eta = [0.8]
    stat_dict = run_and_conclude(hg_file, pl_file, eta, res_path, conclude_file, True, n, draw_density)

    print("plot conclude")
    base_dict = get_baseline()
    val_base = base_dict[design]["value"][k]
    hpwl_base = base_dict[design]["hpwl"][k]
    res_pic_name = f"res/ispd2005/{design}/num/conclude"
    plot_conclude(stat_dict, res_pic_name, val_base, hpwl_base)


def get_baseline():
    base_file = os.path.join("res", "ispd2005", "conclude.hg.origin.json")
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = jstyleson.load(f)
    base_dict = dict()
    for instance, conclude_info in baseline.items():
        design, k = instance.split(".")
        k = int(k)
        if design not in base_dict:
            base_dict[design] = {"value": {}, "hpwl": {}}
        base_dict[design]["value"][k] = conclude_info["value"]
        base_dict[design]["hpwl"][k] = conclude_info["hpwl"]
    return base_dict


def run_and_conclude(
    hg_file, pl_file, eta, res_path, conclude_file, new_conclude=False, n=5, density_func=None
):
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
        hg_list = dataflow_improve_eta(hg, res_path, eta, n, density_func)
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


def generate_one_hg(hg, etai, vir_file, vir_edge, density_func):
    # generate hgi
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
    if density_func:
        density_func(hgi)
    return etai, hgi


def dataflow_improve_eta(hg: DiHypergraph, res_path, eta=[1.0], n=4, density_func=None):
    """
    生成不同加边比例的超图列表
    """
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
    p = Pool(n)
    task_list = []
    for etai in eta:
        vir_file = os.path.join(
            res_path, os.path.basename(hg.hg_src_file.replace(".hg", f".{int(100*etai):02d}.vir"))
        )
        task_list.append(p.apply_async(generate_one_hg, args=(hg, etai, vir_file, vir_edge, density_func)))
    p.close()
    p.join()
    for t in task_list:
        hg_list.append(t.get())
    hg_list.sort()
    hg_list = [hgi for _, hgi in hg_list]
    return hg_list


def par_one_file(hg: DiHypergraph, res_path, eta, new_par=False):
    """
    切分特定的一个超图
    """
    hg_file = hg.hg_file
    print(hg_file, eta)
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
        print(cmd)
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

    stat_file = par_file + ".stat"
    with open(stat_file, "w", encoding="utf8") as f:
        f.write(f"{eta:.4f} {run_time:.4f} {val:.4f}\n")

    stat_key = res_name

    # 将切分结果可视化
    pic = os.path.join(res_path, f"{stat_key}.png")
    plot_pl_with_par(par_dict, pic)

    return stat_key, {
        "value": val,
        "hpwl": hpwl,
        "ncut": len(hpwl_list),
        "run_time": run_time,
        "eta": eta,
    }


# def visualize_results(pl_file, res_path, stat_dict):
#     """'
#     将切分效果最好的结果可视化
#     """
#     for key, _ in stat_dict.items():
#         pl = load_pl(pl_file)
#         par_file = os.path.join(res_path, key)
#         par_list = load_par(par_file)
#         par_dict = generate_par(par_list, pl)
#         pic = os.path.join(res_path, f"{key}.png")
#         plot_pl_with_par(par_dict, pic)


def plot_conclude(stat_dict, res_pic_name, val_base=None, hpwl_base=None):
    """
    统计切分结果，并可视化
    """
    eval_list = [(v["eta"], v["value"], v["hpwl"]) for v in stat_dict.values()]
    eval_list.sort()
    if val_base is None:
        val_base = eval_list[0][1]
    eta, val, hpwl = [], [], []
    for etai, vali, hpwli in eval_list:
        eta.append(etai)
        val.append((vali - val_base) / val_base if val_base else vali)
        hpwl.append(-(hpwli - hpwl_base) / hpwl_base if hpwl_base else hpwli)

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
    ax.scatter(eta, val, c="b")
    fig.savefig(res_pic_name + ".value.png", dpi=300)
    plt.close(fig)

    # plot hpwl
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel(
        r"$-\frac{hpwl_{vir}-hpwl_{hg}}{hpwl_{hg}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax2.scatter(eta, hpwl, c="r")
    fig2.savefig(res_pic_name + ".hpwl.png", dpi=300)
    plt.close(fig2)


def run_all():
    num_value("adaptec1")
    num_value("adaptec2")
    num_value("adaptec3")
    num_value("adaptec4")
    num_value("bigblue1")
    num_value("bigblue2")


if __name__ == "__main__":
    # stable("adaptec1")
    import sys

    # design = "adaptec1"
    # if len(sys.argv) > 1:
    #     design = sys.argv[1]
    # num_value(design)

    run_all()
