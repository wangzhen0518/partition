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

"""
以不同比例删边，观察切分效果
"""


def run_once(config, tag, is_vis, n=8):
    # 读入 hypergraph
    print("read hypergraph")
    task_lst = []
    pool = Pool(n)
    design_lst = config["design"]
    for design in design_lst:
        task_lst.append(pool.apply_async(load_design, args=(design, tag)))
    pool.close()
    pool.join()

    g_list = []
    for task in task_lst:
        g = task.get()
        g_list.append(g)

    # TODO 改成管道/生产者-消费者形式，使 partition 不等待 hg 全部生成完
    print("start partition")
    pool = Pool(n)
    task_lst = []
    stat_dict = dict()
    ubf = config["KaHyPar"]["UBfactor"][0]
    for g in g_list:
        g: DiGraph
        result_pth = os.path.join("res", "ispd2005", g.design, tag)
        os.system(f"mkdir -p {result_pth}")
        for k in config["KaHyPar"]["k"]:
            # stat_key, stat_info = run_partition(hg, k, ubf, method_pth, use_vir, is_vis)
            # stat_dict[stat_key] = stat_info

            task_lst.append(pool.apply_async(run_partition, args=(g, k, ubf, result_pth, is_vis, False)))
    pool.close()
    pool.join()
    for task in task_lst:
        stat_key, stat_info = task.get()
        stat_dict[stat_key] = stat_info

    print("conclude")
    conclude_file = os.path.join("res", "ispd2005", f"conclude.delete_add.{tag}.json")
    print(conclude_file)
    # if not os.path.exists(conclude_file):
    with open(conclude_file, "w", encoding="utf-8") as f:
        jstyleson.dump(stat_dict, f, sort_keys=True, indent=4)
    return stat_dict


def generate_hg(design, new_hg=False):
    hg = DiHypergraph()
    pl_file = os.path.join("res", "ispd2005", design, f"{design}.gp.pl")
    hg.read_pl(pl_file)
    hg_file = os.path.join("res", "ispd2005", design, f"{design}.hg.dire")
    if os.path.exists(hg_file) and not new_hg:
        hg.read_from_file(hg_file)
    else:
        pl_config = os.path.join("pl_config", 'ispd2005', design + ".json")
        hg.build_from_config(pl_config, hg_file)
    return hg


def generate_g(design, new_g=False):
    g = DiGraph()
    pl_file = os.path.join("res", "ispd2005", design, f"{design}.gp.pl")
    g.read_pl(pl_file)
    hg = generate_hg(design)
    g_file = os.path.join("res", "ispd2005", design, f"{design}.80.hg.dire")
    if os.path.exists(g_file) and not new_g:
        g.read_from_file(g_file)
        g.hg = hg
        g.pl = copy.deepcopy(hg.pl)
    else:

        g.read_from_hg(hg, 0.8)
        g.write_dire(g_file)
        g.write(g_file.replace(".dire", ""))
    return g


def add_vir_edge(g: DiGraph, tag: str):
    vir_path = os.path.join("res", "ispd2005", g.design, tag)
    os.system(f"mkdir -p {vir_path}")
    vir_g_file = os.path.join(vir_path, os.path.basename(g.g_src_file).replace(".hg", f".{tag}.vir"))
    if os.path.exists(vir_g_file):
        g.read_from_file(vir_g_file)
    else:
        vir_edge_file = os.path.join(vir_path, f"vir_edge.{tag}.bin")
        if os.path.exists(vir_edge_file):
            with open(vir_edge_file, "rb") as f:
                vir_edge = pk.load(f)
        else:
            vir_edge = g.cal_dataflow(k=3, w_thre=5 * g.aver_weight)
            with open(vir_edge_file, "wb") as f:
                pk.dump(vir_edge, f)
        g.add_vir_edge(vir_edge)
        g.write_dire(vir_g_file)
        g.write(vir_g_file.replace(".dire", ""))


def load_design(design, tag):
    print(design)
    os.system(f"mkdir -p {os.path.join('res','ispd2005',design)}")
    g = generate_g(design, new_g=False)
    add_vir_edge(g, tag)
    return g


def run_partition(g: DiGraph, k, ubf, result_pth, is_vis=False, new_par=False):
    par_file = os.path.join(result_pth, del_ext_name(g.g_file) + f".{k}")
    res_file = par_file + ".res"
    # 处理运行结果
    if os.path.exists(par_file) and os.path.exists(res_file) and not new_par:
        print(f"{par_file}.res exists")
        with open(par_file + ".res", encoding="utf-8") as f:
            res = f.read()
    else:
        cmd = f"KaHyPar -h {g.g_file} -k {k} -e {ubf} -o km1 -m direct -p ./kahypar_config/km1_kKaHyPar_sea20.ini -w 1"
        print(cmd)
        status, res = subprocess.getstatusoutput(cmd)
        if status == 0:
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
    # 生成可视化图片
    if is_vis:
        vis_file = par_file + f".png"
        plot_pl_with_par(par_dict, vis_file)
    stat_key = f"{g.design}.{k}"
    return stat_key, {
        "value": val,
        "value_list": val_list,
        "hpwl": hpwl,
        "ncut": len(hpwl_list),
        # "hpwl_list": hpwl_list,
        "run_time": run_time,
    }


if __name__ == "__main__":
    benchmark = "ispd2005"
    b_pth = os.path.join("res", benchmark)
    os.system(f"mkdir -p {b_pth}")
    config_file = os.path.join("par_config", benchmark, "config.json")
    with open(config_file, encoding="utf-8") as f:
        config = jstyleson.load(f)
    num_thread = 8
    stat_info = run_once(config, "base", is_vis=True, n=num_thread)
    # stat_info = run_once(config, "shrink", is_vis=True, n=num_thread)
    # stat_info = run_once(config, "shrink1", is_vis=True, n=num_thread)
    # stat_info = run_once(config, "enlarge", is_vis=True, n=num_thread)
    stat_info = run_once(config, "enlarge10000", is_vis=True, n=num_thread)
    # with open("res/ispd2005/conclude.delete_edge.json") as f:
    #     stat_info = jstyleson.load(f)
    # draw_conclude(stat_info)
