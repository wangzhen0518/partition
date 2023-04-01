import os
import glob
import subprocess
import matplotlib.pyplot as plt
import json

from multiprocessing import Pool


from utils import visualize_graph, del_ext_name
from partition_method import task, default_method


class config_info:
    def __init__(self, benchmark, is_vis, method: default_method):
        """
        根据benchmark, 从test/${benchmark_name}/config.json读出运行该benchmark的配置
        如果没有test/${benchmark_name}/config.json文件，则读取test/default_config.json，使用默认配置

        config_dict: dict(
            'hg_pth': ..., # 超图文件路径
            'par_pth': ..., # 切分文件存储路径
            'vis_pth': ..., # 可视化结果存储路径
            'stats_pth': ..., # 切分
            'k': [...],
            'UBfactor': [...],
            other args ...,
        )
        """

        self.task_list = method.get_task_list()

        config_file = os.path.join("test", benchmark, "config.json")
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as f:
                self.config_dict = json.loads(f)
        else:
            config_file = os.path.join("test", "default_config.json")
            with open(config_file, encoding="utf-8") as f:
                self.config_dict = json.loads(f)
        # add *_pth configs
        if isinstance(method, str):
            method_name = method
        else:  # method is a function or class
            method_name = method.__name__
        self.config_dict["hg_pth"] = os.path.join("benchmark", benchmark, "hypergraph")
        self.config_dict["par_pth"] = os.path.join("res", benchmark, method_name, "par")
        self.config_dict["vis_pth"] = os.path.join("res", benchmark, method_name, "vis")
        self.config_dict["stats_pth"] = os.path.join("res", benchmark, method_name, "stats")
        self.method = method
        self.is_vis = is_vis


class partition_runner:
    def __init__(self, method: default_method, n=4):
        """
        benchmark
        method: a class
        is_vis: 是否进行可视化绘图
        n: 多进程数
        """
        self.n = n
        self.task_list = method.get_task_list()

        # self.hg_list = self.__get_hg_file()

    def run(self):
        pool = Pool(self.n)
        for t in self.task_list:
            pool.apply_async(t.run)
        pool.close()
        pool.join()

    def __get_hg_file(self):
        # return [
        #     f"{self.__hg_pth}/adaptec3.hg",
        #     f"{self.__hg_pth}/adaptec4.hg",
        #     f"{self.__hg_pth}/bigblue2.hg",
        # ]
        return glob.glob(f"{self.hg_pth}/*.hg")

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
                hmetis  HGraphFile [FixFile] Nparts UBfactor Nruns CType RType Vcycle Reconst dbglvl
                """
                if isinstance(self.__method, str):
                    if self.__method == "shmetis":
                        status, res = subprocess.getstatusoutput(f"shmetis {file} {k} {UBfactor}")
                    elif self.__method == "hmetis":
                        status, res = subprocess.getstatusoutput(f"hmetis {file} {k} {UBfactor}")
                    else:
                        raise ValueError(f"Error Command: {self.__method}")
                else:
                    status, res = self.__method(file, k, UBfactor)
                # status_list.append(status)
                # res_list.append(res)
                stats_file = f"{del_ext_name(file)}.part.{k}"
                self.__save_stats(res, stats_file)

        return status_list, res_list

    def __move_par_results(self):
        status, res = subprocess.getstatusoutput(
            f'mv {self.hg_pth}/*.part.* {self.__par_pth}/; rename "s/.hg//" {self.__par_pth}/*'
        )
        return status, res

    def __visual_results(self):
        gfile_list = self.__get_hg_file()
        for gfile in gfile_list:
            pfile_list = self.__get_par_file(gfile)
            for pfile in pfile_list:
                visualize_graph(gfile, pfile, self.__vis_pth)

    def set_hg_pth(self, hg_pth):
        self.hg_pth = hg_pth

    def set_par_pth(self, par_pth):
        self.__par_pth = par_pth

    def set_vis_pth(self, vis_pth):
        self.__vis_pth = vis_pth

    def set_stats_pth(self, stats_pth):
        self.__stats_pth = stats_pth

    def set_method(self, method):
        self.__method = method
