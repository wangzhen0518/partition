import glob
import os
import subprocess
from pathlib import Path

from utils import generate_single_hg_file
from runner import partition_runner


def generate_hg_file(src_dir, dst_dir):
    """
    循环使用placedb读取并处理输入文件
    """
    subprocess.getstatusoutput(f"mkdir -p {dst_dir}")
    file_list = glob.glob(f"{src_dir}/*.json")
    # file_list = [f"{src_dir}/adaptec3.json", f"{src_dir}/adaptec4.json", f"{src_dir}/bigblue2.json"]
    for file in file_list:
        generate_single_hg_file(file, os.path.join(dst_dir, Path(file).stem + ".hg"))


if __name__ == "__main__":
    config_pth = "./test/ispd2005"
    hg_pth = "./benchmarks/hypergraph/ispd2005"
    par_pth = "./benchmarks/res/ispd2005/par"
    vis_pth = "./benchmarks/res/ispd2005/vis"
    stats_pth = "./benchmarks/res/ispd2005/stats"

    # hg_pth = "./benchmarks/hypergraph"
    # par_pth = "./benchmarks/res/par"
    # vis_pth = "./benchmarks/res/vis"
    # stats_pth = "./benchmarks/res/stats"

    # generate_hg_file(config_pth, hg_pth)
    runner = partition_runner(hg_pth, par_pth, vis_pth, stats_pth, "shmetis")
    runner.run()
