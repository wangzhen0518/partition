import os
import jstyleson
import subprocess
import numpy as np
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool

from hypergraph import DiHypergraph, DiGraph
from manager import generate_par, eval_par, eval_par_HPWL,eval_par_weight
from utils import plot_pl_with_par, del_ext_name, analysis_stats, load_par
from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB

"""
以不同比例删边，观察切分效果
"""

eta = np.linspace(0, 1, 21)[1:]


def delete_one_hg(hg: DiHypergraph, eta):
    print(f"{eta:.3f}", hg.hg_file)
    g = DiGraph()
    g_file = os.path.basename(hg.hg_src_file).replace("hg", f"{int(100*eta)}.hg")
    g_file = os.path.join(os.path.dirname(hg.hg_src_file), "delete_edge", g_file)
    if os.path.exists(g_file):
        g.read_from_file(g_file)
        g.pl = copy.deepcopy(hg.pl)
        g.hg = hg
    else:
        g.read_from_hg(hg, eta)
        g.write_dire(g_file)
        g.write(g_file.replace(".dire", ""))
    return eta, g


def load_design(benchmark, design, b_pth, n=8):
    print(design)
    design_pth = os.path.join(b_pth, design)
    os.system(f"mkdir -p {design_pth}")
    hg = DiHypergraph()
    pl_file = os.path.join(design_pth, design + ".gp.pl")
    hg.read_pl(pl_file)
    hg_file = os.path.join(design_pth, design + ".hg.dire")
    if os.path.exists(hg_file):
        hg.read_from_file(hg_file)
    else:
        # 无论是否使用 virtual edge, 都需要确保 .hg 文件存在
        hg_ori_file = os.path.join(design_pth, design + ".hg.dire")
        if not os.path.exists(hg_ori_file):
            pl_config = os.path.join("pl_config", benchmark, design + ".json")
            hg.build_from_config(pl_config, hg_ori_file)

    os.system(f"mkdir -p {design_pth}/delete_edge")
    p = Pool(n)
    task_list = []
    for etai in eta:
        # delete_one_hg(hg, etai)
        task_list.append(p.apply_async(delete_one_hg, args=(hg, etai)))
    p.close()
    p.join()

    g_list = []
    for t in task_list:
        etai, g = t.get()
        g_list.append((etai, g))
    return g_list


def run_once(benchmark, b_pth, config, tag, is_vis, n=8):
    # 读入 hypergraph
    print("read hypergraph")
    task_lst = []
    pool = Pool(n)
    # g_list = []
    # design_lst = config["design"]
    design = config["design"][0]
    # for design in design_lst:
    g_list = load_design(benchmark, design, b_pth, n)
    # hg_lst.append(hg)
    # task_lst.append(pool.apply_async(load_design, args=(benchmark, design, b_pth, m_type, use_vir, True)))
    # pool.close()
    # pool.join()
    # for task in task_lst:
    #     hg = task.get()
    #     hg_list.append(hg)
    # TODO 改成管道/生产者-消费者形式，使 partition 不等待 hg 全部生成完
    print("start partition")
    pool = Pool(n)
    task_lst = []
    stat_dict = dict()
    ubf = config["KaHyPar"]["UBfactor"][0]
    for etai, g in g_list:
        g: DiGraph
        result_pth = os.path.join(g.design_pth)
        os.system(f"mkdir -p {result_pth}")
        for k in config["KaHyPar"]["k"]:
            # stat_key, stat_info = run_partition(hg, k, ubf, method_pth, use_vir, is_vis)
            # stat_dict[stat_key] = stat_info

            task_lst.append(pool.apply_async(run_partition, args=(g, k, ubf, result_pth, etai, is_vis)))
    pool.close()
    pool.join()
    for task in task_lst:
        stat_key, stat_info = task.get()
        stat_dict[stat_key] = stat_info

    print("conclude")
    conclude_file = os.path.join(b_pth, f"conclude.{tag}.json")
    print(conclude_file)
    # if not os.path.exists(conclude_file):
    with open(conclude_file, "w", encoding="utf-8") as f:
        jstyleson.dump(stat_dict, f, sort_keys=True, indent=4)
    return stat_dict


def run_partition(g: DiGraph, k, ubf, result_pth, eta, is_vis=False, new_par=False):
    par_file = os.path.join(result_pth, del_ext_name(g.g_file) + f".{int(100*eta)}.{k}")
    res_file = par_file + ".res"
    # 处理运行结果
    if os.path.exists(par_file) and os.path.exists(res_file) and not new_par:
        print(f"{par_file}.res exists")
        with open(par_file + ".res", encoding="utf-8") as f:
            res = f.read()
    else:
        # cmd = f"KaHyPar {hg.hg_file} {k} {ubf}"
        cmd = f"KaHyPar -h {g.g_file} -k {k} -e {ubf} -o km1 -m direct -p ./kahypar_config/km1_kKaHyPar_sea20.ini -w 1"
        print(cmd)
        status, res = subprocess.getstatusoutput(cmd)
        if status == 0:
            # subprocess.getstatusoutput(f"mv {hg.hg_file}.part.{k} {par_file}")
            subprocess.getstatusoutput(f"mv {g.g_file}.part{k}.epsilon{ubf}.seed-1.KaHyPar {par_file}")
            with open(par_file + ".res", "w", encoding="utf-8") as f:
                f.write(res)
        else:
            raise RuntimeError(f"{g.g_file}\tError {res}")
    # 处理统计信息
    run_time = analysis_stats(res)
    par_list = load_par(par_file)
    par_dict = generate_par(par_list, g.pl)
    val, val_list = eval_par(par_dict)
    hpwl, hpwl_list = eval_par_HPWL(par_list, g.hg)
    weight,weight_list=eval_par_weight(par_list,g.hg)
    # 生成可视化图片
    if is_vis:
        vis_file = par_file + f".png"
        plot_pl_with_par(par_dict, vis_file)
    stat_key = g.design + f".{int(100*eta)}.{k}"
    return stat_key, {
        "value": val,
        "value_list": val_list,
        "hpwl": hpwl,
        "ncut": len(hpwl_list),
        "weight":weight,
        # "hpwl_list": hpwl_list,
        "run_time": run_time,
    }


def draw_conclude(stat_info):
    base_file = os.path.join("res", "ispd2005", "conclude.hg.origin.json")
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = jstyleson.load(f)

    value_list, hpwl_list, ncut_list = [], [], []
    hpwl_total = []
    weight_list=[]
    k_list = []
    os.system("mkdir -p res/ispd2005/conclude")
    res_pic_name = os.path.join("res", "ispd2005", "conclude", f"conclude.delete_edge")
    for bm, info in stat_info.items():
        bm = bm.split(".")
        k_list.append(int(bm[1]) / 100)
        bm = ".".join([bm[0], bm[2]])
        value_list.append((info["value"] - baseline[bm]["value"]) / baseline[bm]["value"])
        hpwl_list.append(-(info["hpwl"] - baseline[bm]["hpwl"]) / baseline[bm]["hpwl"])
        weight_list.append((info["weight"] - baseline[bm]["weight"]) / baseline[bm]["weight"])
        ncut_list.append((info["ncut"] - baseline[bm]["ncut"]) / baseline[bm]["ncut"])
        hpwl_base = baseline[bm]["hpwl"] * baseline[bm]["ncut"]
        hpwl = info["hpwl"] * info["ncut"]
        hpwl_total.append(-(hpwl - hpwl_base) / hpwl_base)
        
        # plot value
        fig, ax = plt.subplots()
        ax.set_title("value")
        ax.set_ylabel(
            r"$\frac{value_{vir}-value_{hg}}{value_{hg}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax.scatter(k_list, value_list, c="b")
        fig.savefig(res_pic_name + ".value.png", dpi=300)
        plt.close(fig)

        # plot hpwl
        fig2, ax2 = plt.subplots()
        ax2.set_title('hpwl')
        ax2.set_ylabel(
            r"$-\frac{hpwl_{vir}-hpwl_{hg}}{hpwl_{hg}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax2.scatter(k_list, hpwl_list, c="r")
        fig2.savefig(res_pic_name + ".hpwl.png", dpi=300)
        plt.close(fig2)

        # plot ncut
        fig3, ax3 = plt.subplots()
        ax3.set_title('ncut')
        ax3.set_ylabel(
            r"$\frac{ncut_{vir}-ncut_{hg}}{ncut_{hg}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax3.scatter(k_list, ncut_list, c="c")
        fig3.savefig(res_pic_name + ".ncut.png", dpi=300)
        plt.close(fig3)
        
        fig4, ax4 = plt.subplots()
        ax4.set_title("total hpwl")
        ax4.set_ylabel(
            r"-$\frac{hpwl'_{vir}-hpwl'_{hg}}{hpwl'_{hg}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax4.scatter(k_list, hpwl_total, c="b")
        fig4.savefig(res_pic_name + ".hpwl_total.png", dpi=300)
        plt.close(fig4)
        
        # fig5, ax5 = plt.subplots()
        # ax5.set_title("weight")
        # ax5.set_ylabel(
        #     r"$\frac{weight_{vir}-weight_{hg}}{weight_{hg}}$",
        #     fontdict={"size": 16},
        #     rotation=0,
        #     loc="top",
        #     labelpad=-90,
        # )
        # ax5.scatter(k_list, weight_list, c="b")
        # fig5.savefig(res_pic_name + ".hpwl_total.png", dpi=300)
        # plt.close(fig5)

if __name__ == "__main__":
    benchmark = "ispd2005"
    b_pth = os.path.join("res", benchmark)
    os.system(f"mkdir -p {b_pth}")
    config_file = os.path.join("par_config", benchmark, "config.json")
    with open(config_file, encoding="utf-8") as f:
        config = jstyleson.load(f)
    num_thread = 8
    stat_info = run_once(benchmark, b_pth, config, "delete_edge", is_vis=True, n=num_thread)
    # with open("res/ispd2005/conclude.delete_edge.json") as f:
    #     stat_info = jstyleson.load(f)
    draw_conclude(stat_info)
