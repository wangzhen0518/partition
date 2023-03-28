import os
import glob
import subprocess
import matplotlib.pyplot as plt

from utils import visualize_graph

# TODO: 改写成一个类，*_pth作为类属性，后续对于不同 partition 方法采取不同的运行方法、分析方法、数据存放位置


class partition_runner:
    def __init__(self, hg_pth, par_pth, vis_pth, stats_pth, method):
        self.__hg_pth = hg_pth
        self.__par_pth = par_pth
        self.__vis_pth = vis_pth
        self.__stats_pth = stats_pth
        self.__method = method

    def __get_hg_file(self):
        return glob.glob(f"{self.__hg_pth}/*")

    def __get_par_file(self, hg_src):
        """
        返回给定 hypergraph 对应的所有切分结果
        """
        hg_file = os.path.basename(hg_src)
        return glob.glob(f"{self.__par_pth}/{hg_file}.part.*")

    def __save_stats(self, stats, file):
        with open(f"{self.__stats_pth}/{file}", "w", encoding="utf-8") as f:
            f.write(stats)

    def __partition(self):
        k_list = [2, 3, 4]
        UBfactor = 2
        status_list = []
        res_list = []

        file_list = self.__get_hg_file()
        for file in file_list:
            for k in k_list:
                """
                shmetis HGraphFile [FixFile] Nparts UBfactor
                """
                if isinstance(self.__method, str):
                    status, res = subprocess.getstatusoutput(f"{self.__method} {file} {k} {UBfactor}")
                else:
                    status, res = self.__method(file, k, UBfactor)
                # status_list.append(status)
                # res_list.append(res)
                stats_file = f"{os.path.basename(file)}.part.{k}"
                self.__save_stats(res, stats_file)

        return status_list, res_list

    def __move_par_results(self):
        status, res = subprocess.getstatusoutput(f"mv {self.__hg_pth}/*.part.* {self.__par_pth}/")
        return status, res

    def __visual_results(self):
        gfile_list = self.__get_hg_file()
        for gfile in gfile_list:
            pfile_list = self.__get_par_file(gfile)
            for pfile in pfile_list:
                visualize_graph(gfile, pfile, self.__vis_pth)

    def set_hg_pth(self, hg_pth):
        self.__hg_pth = hg_pth

    def set_par_pth(self, par_pth):
        self.__par_pth = par_pth

    def set_vis_pth(self, vis_pth):
        self.__vis_pth = vis_pth

    def set_stats_pth(self, stats_pth):
        self.__stats_pth = stats_pth

    def set_method(self, method):
        self.__method = method

    def run(self):
        subprocess.getstatusoutput(f"mkdir -p {self.__par_pth} {self.__vis_pth} {self.__stats_pth}")
        self.__partition()
        self.__move_par_results()
        self.__visual_results()


if __name__ == "__main__":
    hg_pth = "./benchmarks/hypergraph"
    par_pth = "./benchmarks/res/par"
    vis_pth = "./benchmarks/res/vis"
    stats_pth = "./benchmarks/res/stats"
    runner = partition_runner(hg_pth, par_pth, vis_pth, stats_pth, "shmetis")
    runner.run()
