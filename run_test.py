import os
import glob
import subprocess
import matplotlib.pyplot as plt

from utils import visualize_graph


def get_hg_file(gsrc):
    return glob.glob(f"{gsrc}/*")


def get_par_file(gsrc, psrc):
    """
    返回给定 hypergraph 对应的所有切分结果
    """
    hg_file = os.path.basename(gsrc)
    return glob.glob(f"{psrc}/{hg_file}.part.*")


def partition(src):
    k_list = [2, 3, 4]
    UBfactor = 2
    # filelist = os.listdir("./benchmarks/hypergraph")
    file_list = get_hg_file(src)
    status_list = []
    res_list = []
    for file in file_list:
        for k in k_list:
            status, res = subprocess.getstatusoutput(f"shmetis {file} {k} {UBfactor}")
            status_list.append(status)
            res_list.append(res)
    """
    shmetis HGraphFile [FixFile] Nparts UBfactor
    """
    return status_list, res_list


def move_par_results(src, dst):
    status, res = subprocess.getstatusoutput(f"mkdir -p {dst}; mv {src}/*.part.* {dst}/")
    return status, res


def visual_results(gsrc, psrc, dst):
    gfile_list = get_hg_file(gsrc)
    for gfile in gfile_list:
        pfile_list = get_par_file(gfile, psrc)
        for pfile in pfile_list:
            visualize_graph(gfile, pfile, dst)


if __name__ == "__main__":
    hg_pth = "./benchmarks/hypergraph"
    par_pth = "./benchmarks/res/par"
    vis_pth = "./benchmarks/res/vis"
    partition(hg_pth)
    move_par_results(hg_pth, par_pth)
    visual_results(hg_pth, par_pth, vis_pth)
