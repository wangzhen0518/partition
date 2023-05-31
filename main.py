import os
import jstyleson
import subprocess
import numpy as np

from multiprocessing import Pool

from hypergraph import DiHypergraph
from manager import generate_par, eval_par, eval_par_HPWL
from utils import plot_pl_with_par, del_ext_name, analysis_stats, load_par
from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB

pl_ext = "gp"  # gp or ntup


def load_design(benchmark, design, b_pth, m_type, use_vir=True, new_hg=False):
    print(design)
    hg_ext = ".vir" if use_vir else ".hg"
    design_pth = os.path.join(b_pth, design)
    os.system(f"mkdir -p {design_pth}")
    hg = DiHypergraph()
    hg_file = os.path.join(design_pth, design + hg_ext + ".dire")
    if os.path.exists(hg_file) and not new_hg:
        hg.read_from_file(hg_file)
    else:
        # 无论是否使用 virtual edge, 都需要确保 .hg 文件存在
        hg_ori_file = os.path.join(design_pth, design + ".hg" + ".dire")
        # if not os.path.exists(hg_ori_file):
        pl_config = os.path.join("pl_config", benchmark, design + ".json")
        hg.build_from_config(pl_config, hg_ori_file)
        # else:
        #     hg.read_from_file(hg_ori_file)
        # 如果使用 virtual edge, 并且 .vir 存在, 那么上一个 if 已经读过文件
        # 所以此处为使用 virtual edge, 但是 .vir 文件不存在, 应生成 .vir 文件
        if use_vir:
            hg.dataflow_improve(m_type)
    pl_file = os.path.join(design_pth, design + f".{pl_ext}.pl")
    hg.read_pl(pl_file)
    return hg


def run_partition(hg: DiHypergraph, k, ubf, result_pth, use_vir=True, is_vis=False, new_par=False):
    hg_ext = "vir" if use_vir else "hg"
    par_file = os.path.join(result_pth, hg.design + f".{hg_ext}.{k}")
    res_file = par_file + ".res"
    # 处理运行结果
    if os.path.exists(par_file) and os.path.exists(res_file) and not new_par:
        print(f"{par_file}.res exists")
        with open(par_file + ".res", encoding="utf-8") as f:
            res = f.read()
    else:
        # cmd = f"KaHyPar {hg.hg_file} {k} {ubf}"
        cmd = f"KaHyPar -h {hg.hg_file} -k {k} -e {ubf} -o km1 -m direct -p ./kahypar_config/km1_kKaHyPar_sea20.ini -w 1"
        print(cmd)
        status, res = subprocess.getstatusoutput(cmd)
        if status == 0:
            # subprocess.getstatusoutput(f"mv {hg.hg_file}.part.{k} {par_file}")
            subprocess.getstatusoutput(f"mv {hg.hg_file}.part{k}.epsilon{ubf}.seed-1.KaHyPar {par_file}")
            with open(par_file + ".res", "w", encoding="utf-8") as f:
                f.write(res)
        else:
            raise RuntimeError(f"{hg.hg_file}\tError {res}")
    # 处理统计信息
    run_time = analysis_stats(res)
    par_list = load_par(par_file)
    par_dict = generate_par(par_list, hg.pl)
    val, val_list = eval_par(par_dict)
    hpwl, hpwl_list = eval_par_HPWL(par_list, hg)
    # 生成可视化图片
    if is_vis:
        vis_file = par_file + f".{pl_ext}.pdf"
        plot_pl_with_par(par_dict, vis_file)
    stat_key = hg.design + f".{k}"
    return stat_key, {
        "value": val,
        "value_list": val_list,
        "hpwl": hpwl,
        "ncut": len(hpwl_list),
        # "hpwl_list": hpwl_list,
        "run_time": run_time,
    }


def run_once(benchmark, b_pth, config, m_type, use_vir, is_vis, n=8):
    # 读入 hypergraph
    print("read hypergraph")
    task_lst = []
    pool = Pool(n)
    hg_lst = []
    design_lst = config["design"]
    for design in design_lst:
        # hg = load_design(benchmark, design, b_pth, use_vir)
        # hg_lst.append(hg)
        task_lst.append(pool.apply_async(load_design, args=(benchmark, design, b_pth, m_type, use_vir, False)))
    pool.close()
    pool.join()
    for task in task_lst:
        hg = task.get()
        hg_lst.append(hg)
    # TODO 改成管道/生产者-消费者形式，使 partition 不等待 hg 全部生成完
    print("start partition")
    pool = Pool(n)
    task_lst = []
    stat_dict = dict()
    ubf = config["KaHyPar"]["UBfactor"][0]
    for hg in hg_lst:
        hg: DiHypergraph
        result_pth = os.path.join(hg.design_pth, m_type)
        os.system(f"mkdir -p {result_pth}")
        for k in config["KaHyPar"]["k"]:
            # stat_key, stat_info = run_partition(hg, k, ubf, method_pth, use_vir, is_vis)
            # stat_dict[stat_key] = stat_info

            task_lst.append(
                pool.apply_async(run_partition, args=(hg, k, ubf, result_pth, use_vir, is_vis, False))
            )
    pool.close()
    pool.join()
    for task in task_lst:
        stat_key, stat_info = task.get()
        stat_dict[stat_key] = stat_info

    print("conclude")
    hg_ext = "vir" if use_vir else "hg"
    conclude_file = os.path.join(b_pth, f"conclude.{hg_ext}.{m_type}.json")
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
    num_thread = 8

    # run_once(benchmark, b_pth, config, "base", use_vir=True, is_vis=True, n=num_thread)
    # run_once(benchmark, b_pth, config, "enlarge", use_vir=True, is_vis=True, n=num_thread)
    # run_once(benchmark, b_pth, config, "shrink", use_vir=True, is_vis=True, n=num_thread)
    # run_once(benchmark, b_pth, config, "enlarge10000", use_vir=True, is_vis=True, n=num_thread)
    # run_once(benchmark, b_pth, config, "shrink1", use_vir=True, is_vis=True, n=num_thread)
    run_once(benchmark, b_pth, config, "origin", use_vir=False, is_vis=True, n=num_thread)
