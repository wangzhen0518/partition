import os
import jstyleson
import glob
import subprocess
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp


from abc import ABC, abstractmethod
from multiprocessing import Pool
from datetime import datetime

from utils import del_ext_name, plot_pl_with_par
from manager import generate_par, eval_par


class default_method(ABC):
    def __init__(self, benchmark, is_vis=False):
        """
        benchmark: 测试集
        n: 多进程数
        """
        self.__name__ = self.get_name()
        self.logger = None
        self.is_vis = None
        self.benchmark = None
        self.stats_dict = None
        self.hg_pth = None
        self.pl_pth = None
        self.par_pth = None
        self.vis_pth = None
        self.stats_pth = None
        self.config_dict = None
        self.set_config(benchmark, is_vis)

    @classmethod
    def get_name(cls):
        return cls.__name__

    def get_hg_files(self):
        hg_file_list = glob.glob(os.path.join(self.hg_pth, "*.hg"))
        hg_file_list.sort()
        return ["res/ispd2005/hypergraph/adaptec1.hg.vir"]
        return hg_file_list

    def set_logger(self):
        self.logger = logging.getLogger(self.__name__)
        # 创建一个handler，用于写入日志文件
        log_name = os.path.join(
            "res", self.benchmark, self.__name__, f"{datetime.now().date()}_{self.__name__}.log"
        )
        fh = logging.FileHandler(log_name, mode="w+", encoding="utf-8")
        # 再创建一个handler用于输出到控制台
        ch = logging.StreamHandler()
        # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        # 定义日志输出层级
        self.logger.setLevel(logging.DEBUG)
        # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
        fh.setFormatter(fmt)
        # 为控制台操作符绑定格式（可以绑定多种格式例ch.setFormatter(formatter2)）
        ch.setFormatter(fmt)
        # 给logger对象绑定文件操作符
        self.logger.addHandler(fh)
        # 给logger对象绑定文件操作符
        self.logger.addHandler(ch)
        return self.logger

    def set_config(self, benchmark, is_vis):
        self.is_vis = is_vis
        self.benchmark = benchmark
        self.stats_dict = mp.Manager().dict()
        self.hg_pth = os.path.join("res", benchmark, "hypergraph")
        self.pl_pth = os.path.join("res", benchmark, "pl")
        self.par_pth = os.path.join("res", benchmark, self.__name__ + "_ori", "par")
        self.vis_pth = os.path.join("res", benchmark, self.__name__ + "_ori", "vis")
        self.stats_pth = os.path.join("res", benchmark, self.__name__ + "_ori", "stats")
        # self.par_pth = os.path.join("res", benchmark, self.__name__, "par")
        # self.vis_pth = os.path.join("res", benchmark, self.__name__, "vis")
        # self.stats_pth = os.path.join("res", benchmark, self.__name__, "stats")
        subprocess.getstatusoutput(f"mkdir -p {self.par_pth} {self.vis_pth} {self.stats_pth}")
        self.set_logger()
        self.load_config()

    def load_config(self):
        config_file = os.path.join("test", self.benchmark, "config.json")
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as f:
                self.config_dict = jstyleson.load(f)
        else:
            config_file = os.path.join("test", "default_config.json")
            with open(config_file, encoding="utf-8") as f:
                self.config_dict = jstyleson.load(f)

        if self.__name__ in self.config_dict.keys():
            self.config_dict = self.config_dict[self.__name__]
        else:
            self.config_dict = self.config_dict["default"]

    @abstractmethod
    def get_task_list(self):
        ...

    @abstractmethod
    def run(self, *args):
        ...

    @abstractmethod
    def conclude(self):
        """
        汇总统计数据，以及收尾处理
        """
        ...

    def run_all(self, n=4):
        pool = Pool(n)
        for t in self.get_task_list():
            pool.apply_async(self.run, t)
        pool.close()
        pool.join()
        self.conclude()


class shmetis(default_method):
    def __init__(self, benchmark, is_vis=False):
        super(shmetis, self).__init__(benchmark, is_vis)
        """
        config_dict: dict(
            'k': [...],
            'UBfactor': [...]
        )"""

    def get_task_list(self):
        for hg_file in self.get_hg_files():
            for k in self.config_dict["k"]:
                for ubf in self.config_dict["UBfactor"]:
                    yield [hg_file, k, ubf]

    def partition(self, *args):
        hg_file, k, ubf = args
        state, res = subprocess.getstatusoutput(f"shmetis {hg_file} {k} {ubf}")
        return state, res

    def analysis_stats(self, res: str):
        res = res.split("\n")[-3:-1]
        run_time = float(res[0].split(":")[-1].replace("sec", ""))
        io_time = float(res[1].split(":")[-1].replace("sec", ""))
        return run_time, io_time

    def post_processing(self, args):
        hg_file, k, ubf, res = args
        design = del_ext_name(hg_file)

        # 移动 partition 结果文件
        res_name = design + f".{k}.{ubf}"
        par_ori_file = hg_file + f".part.{k}"
        par_file = os.path.join(self.par_pth, res_name)
        subprocess.getstatusoutput(f"mv {par_ori_file} {par_file}")

        # 分析运行时间
        run_time, io_time = self.analysis_stats(res)

        # 评价切分好坏
        pl_file = os.path.join(self.pl_pth, design + ".gp.pl")
        par_dict = generate_par(par_file, pl_file)
        val, _ = eval_par(par_dict)

        # 记录分析结果
        self.stats_dict[res_name] = {"io": io_time, "run": run_time, "value": val}
        res_file = os.path.join(self.stats_pth, res_name)
        with open(res_file, "w", encoding="utf-8") as f:
            f.write(f"{run_time:.4f} {io_time:.4f} {val:.4f}\n")

        if self.is_vis:
            vis_file = os.path.join(self.vis_pth, res_name + ".png")
            plot_pl_with_par(par_dict, vis_file)

    def run(self, *args):
        # TODO 将 *.hg 文件的生成过程也加到这里，需要修改 self.get_hg_files()
        msg = " ".join([str(s) for s in args])
        self.logger.info(msg + " start")
        state,res=self.partition(args)
        state, res = 0, ""
        if state == 0:
            args = list(args).append(res)
            self.post_processing(args)
        self.logger.info(msg + " end")

    def conclude(self):
        conclude_file = os.path.join(self.stats_pth, "conclude.json")
        with open(conclude_file, "w", encoding="utf-8") as f:
            jstyleson.dump(self.stats_dict.copy(), f, sort_keys=True, indent=4)
