import os
import jstyleson
import subprocess
import numpy as np

from multiprocessing import Pool

from hypergraph import DiHypergraph
from manager import generate_par, eval_par
from utils import plot_pl_with_par, del_ext_name, analysis_stats, load_par
from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB

pl_ext = "gp"  # gp or ntup


def load_design(benchmark, design, b_pth, use_vir=True):
    print(design)
    hg_ext = ".vir" if use_vir else ".hg"
    design_pth = os.path.join(b_pth, design)
    os.system(f"mkdir -p {design_pth}")
    hg = DiHypergraph()
    hg_file = os.path.join(design_pth, design + hg_ext + ".dire")
    if os.path.exists(hg_file):
        hg.read_from_file(hg_file)
    else:
        # 无论是否使用 virtual edge, 都需要确保 .hg 文件存在
        hg_ori_file = os.path.join(design_pth, design + ".hg" + ".dire")
        if not os.path.exists(hg_ori_file):
            pl_config = os.path.join("pl_config", benchmark, design + ".json")
            hg.build_from_config(pl_config, hg_ori_file)
        else:
            hg.read_from_file(hg_ori_file)
        # 如果使用 virtual edge, 并且 .vir 存在, 那么上一个 if 已经读过文件
        # 所以此处为使用 virtual edge, 但是 .vir 文件不存在, 应生成 .vir 文件
        if use_vir:
            hg.dataflow_improve()
    pl_file = os.path.join(design_pth, design + f".{pl_ext}.pl")
    hg.read_pl(pl_file)
    return hg


def run_partition(hg: DiHypergraph, k, ubf, method_pth, use_vir=True, is_vis=False):
    N = 10
    hg_ext = ".vir" if use_vir else ".hg"
    val_lst = []
    run_time_lst = []
    io_time_lst = []
    for i in range(N):
        res_file = os.path.join(method_pth, hg.design + hg_ext + f".{k}.{i}")
        # 处理运行结果
        if not os.path.exists(res_file + ".res"):
            cmd = f"shmetis {hg.hg_file} {k} {ubf}"
            print(i, cmd)
            status, res = subprocess.getstatusoutput(cmd)
            if status == 0:
                os.system(f"mv {hg.hg_file}.part.{k} {res_file}")
                with open(res_file + ".res", "w", encoding="utf-8") as f:
                    f.write(res)
            else:
                print(f"Error {res}")
        else:
            print(f"{res_file}.res exists")
            with open(res_file + ".res", encoding="utf-8") as f:
                res = f.read()
        # 处理统计信息
        par = load_par(res_file)
        par_dict = generate_par(par, hg.pl)
        stat_file = res_file + f".{pl_ext}.stat"
        if not os.path.exists(stat_file):
            run_time, io_time = analysis_stats(res)
            val, _ = eval_par(par_dict)
            with open(stat_file, "w", encoding="utf-8") as f:
                f.write(f"{run_time:.4f} {io_time:.4f} {val:.4f}\n")
        else:
            with open(stat_file, encoding="utf-8") as f:
                run_time, io_time, val = [float(c) for c in f.read().split()]
        stat_key = os.path.basename(res_file).replace(hg_ext, "")
        val_lst.append(val)
        run_time_lst.append(run_time)
        io_time_lst.append(io_time)
        # 生成可视化图片
        if is_vis:
            vis_file = res_file + f".{pl_ext}.png"
            if not os.path.exists(vis_file):
                plot_pl_with_par(par_dict, vis_file)
    val = np.average(val_lst)
    run_time = np.average(run_time_lst)
    io_time = np.average(io_time_lst)
    # idx = np.argmin(val_lst)
    stat_key = ".".join(stat_key.split(".")[:-1])
    return stat_key, {"value": val, "run": run_time, "io": io_time}


def run_once(benchmark, b_pth, config, use_vir, is_vis, n=8):
    # 读入 hypergraph
    print("read hypergraph")
    task_lst = []
    pool = Pool(n)
    hg_lst = []
    design_lst = config["design"]
    for design in design_lst:
        # hg = load_design(benchmark, design, b_pth, use_vir)
        # hg_lst.append(hg)
        task_lst.append(pool.apply_async(load_design, args=(benchmark, design, b_pth, use_vir)))
    pool.close()
    pool.join()
    hg_lst = []
    for task in task_lst:
        hg = task.get()
        hg_lst.append(hg)
    # TODO 改成管道/生产者-消费者形式，使 partition 不等待 hg 全部生成完
    print("start partition")
    pool = Pool(n)
    task_lst = []
    stat_dict = dict()
    ubf = config["shmetis"]["UBfactor"][0]
    for hg in hg_lst:
        hg: DiHypergraph
        method_pth = os.path.join(hg.design_pth, "shmetis")
        os.system(f"mkdir -p {method_pth}")
        for k in config["shmetis"]["k"]:
            task_lst.append(pool.apply_async(run_partition, args=(hg, k, ubf, method_pth, use_vir, is_vis)))
    pool.close()
    pool.join()
    for task in task_lst:
        stat_key, stat_info = task.get()
        stat_dict[stat_key] = stat_info

    print("conclude")
    hg_ext = ".vir" if use_vir else ".hg"
    conclude_file = os.path.join(b_pth, f"conclude{hg_ext}.shmetis.{pl_ext}.json")
    print(conclude_file)
    # if not os.path.exists(conclude_file):
    with open(conclude_file, "w", encoding="utf-8") as f:
        jstyleson.dump(stat_dict, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    benchmark = "ispd2005"
    b_pth = os.path.join("res", benchmark)
    os.system(f"mkdir -p {b_pth}")
    config_file = os.path.join("par_config", benchmark, "config.json")
    with open(config_file, encoding="utf-8") as f:
        config = jstyleson.load(f)
    num_thread = 4
    run_once(benchmark, b_pth, config, use_vir=False, is_vis=True, n=num_thread)
    run_once(benchmark, b_pth, config, use_vir=True, is_vis=True, n=num_thread)
