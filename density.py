import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hypergraph import DiHypergraph
from utils import draw_density

matplotlib.use("Agg")


if __name__ == "__main__":
    design_list = [
        "adaptec1", "adaptec2",
        "adaptec3",
           "adaptec4", "bigblue1", "bigblue2", "bigblue3"
    ]
    # hg_ori_list = []
    hg_list = []

    for design in design_list:
        # hg_ori = DiHypergraph()
        # hg_file = f"res/ispd2005/{design}/{design}.hg.dire"
        # hg_ori.read_from_file(hg_file)

        hg = DiHypergraph()
        vir_file = f"res/ispd2005/{design}/{design}.vir.dire"
        hg.read_from_file(vir_file)

        draw_density(hg)
