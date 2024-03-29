import glob
import os, sys
import subprocess
import logging

from hypergraph import generate_single_hg_file
from utils import del_ext_name

# from runner import partition_runner
from partition_method import default_method, shmetis_method


def generate_hg_file(benchmark):
    """
    循环使用placedb读取并处理输入文件
    """
    config_pth = os.path.join("test", benchmark)
    hg_pth = os.path.join("res", benchmark, "hypergraph")
    subprocess.getstatusoutput(f"mkdir -p {hg_pth}")
    file_list = glob.glob(os.path.join(config_pth, "*.json"))
    file_list.sort()
    # file_list = [f"{src_dir}/adaptec3.json", f"{src_dir}/adaptec4.json", f"{src_dir}/bigblue2.json"]
    for file in file_list:
        if "config.json" in file:
            continue
        try:
            generate_single_hg_file(file, os.path.join(hg_pth, del_ext_name(file) + ".hg"))
        except Exception as e:
            print(f"\n[ERROR] {file}\n{e}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        benchmark = sys.argv[1]
    else:
        benchmark = "ispd2005"
    # generate_hg_file(benchmark)
    shmetis = shmetis_method(benchmark, True)
    shmetis.run_all(4)
