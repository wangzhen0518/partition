import glob
import sys, os
import subprocess
import logging

from pathlib import Path

from utils import generate_single_hg_file, del_ext_name

# from runner import partition_runner
from partition_method import default_method, shmetis_method


def generate_hg_file(src_dir, dst_dir):
    """
    循环使用placedb读取并处理输入文件
    """
    subprocess.getstatusoutput(f"mkdir -p {dst_dir}")
    file_list = glob.glob(f"{src_dir}/*.json")
    try:
        file_list.remove("config.json")
    except ValueError:
        pass
    # file_list = [f"{src_dir}/adaptec3.json", f"{src_dir}/adaptec4.json", f"{src_dir}/bigblue2.json"]
    for file in file_list:
        generate_single_hg_file(file, os.path.join(dst_dir, del_ext_name(file) + ".hg"))


if __name__ == "__main__":
    config_pth = "./test/ispd2005"
    hg_pth = "./benchmarks/ispd2005/hypergraph"
    generate_hg_file(config_pth, hg_pth)
    shmetis = shmetis_method("ispd2005", False)
    shmetis.run_all(8)
