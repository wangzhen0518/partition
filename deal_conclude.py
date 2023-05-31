import numpy as np
import jstyleson
import matplotlib.pyplot as plt
import os
import pandas as pd

from utils import load_pl, load_par, dict_append, del_ext_name, generate_benchmark_dict


def sort_keys(key_lst, type):
    idx_list = []
    k_list = []
    if type == "k":
        tmp = []
        for k in key_lst:
            idx = int(k.split(".")[1])
            tmp.append((idx, k))
        tmp.sort()
        for idx, k in tmp:
            idx_list.append(idx)
            k_list.append(k)
    elif type == "g":
        for k in key_lst:
            idx = k.split(".")[0]
            idx_list.append(idx)
            k_list.append(k)
    return idx_list, k_list


def compare(vir_conclude, hg_conclude, type):
    cmp_f = lambda x, y: (x - y) / y
    idx_list, k_list = sort_keys(vir_conclude.keys(), type)
    val_cmp_lst = []
    hpwl_cmp_list = []
    for t in k_list:
        val_cmp_lst.append(cmp_f(vir_conclude[t]["value"], hg_conclude[t]["value"]))
        hpwl_cmp_list.append(cmp_f(vir_conclude[t]["hpwl"], hg_conclude[t]["hpwl"]))
    return idx_list, k_list, val_cmp_lst, hpwl_cmp_list


def plot_cmp(cmp_lst, idx_list, type, metrics):
    fig, ax = plt.subplots()
    x = list(range(len(idx_list)))
    ax.scatter(x, cmp_lst)
    # ax.scatter(x, cmp_ntup_lst)
    ax.axhline(y=0, c="r", ls="-")
    ax.set_ylabel(
        r"$\frac{metrics_{vir}-metrics_{hg}}{metrics_{hg}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax.set_title(metrics)
    idx_r = idx_list[0]
    x_l = x_r = x[0]  # x_l 是区间左端点, x_r 是区间右端点
    for i, (xi, idx) in enumerate(zip(x, idx_list)):
        if idx != idx_r:
            x_r = x[i - 1]
            ax.axvline(x=(x_r + xi) / 2, c="orange", ls="--")
            ax.annotate(
                text=idx_r,
                xy=((x_l + x_r) / 2, 0.1),
                xytext=((x_l + x_r) / 2 - len(str(idx_r)) / (x_r - x_l) / 2, 0.1),
                c="darkred",
                weight="bold",
            )
            print(idx_r)
            idx_r = idx
            x_l = xi
    x_r = x[-1]
    ax.annotate(
        text=idx_r,
        xy=((x_l + x_r) / 2, 0.1),
        xytext=((x_l + x_r) / 2 - len(str(idx_r)) / (x_r - x_l) / 2, 0.1),
        c="darkred",
        weight="bold",
    )
    print(idx_r)
    fig.savefig(f"res/ispd2005/conclude.hpwl.{type}.{metrics}.png", dpi=300)
    plt.close(fig)
    # print(idx_list)


def plot_improve(design_lst):
    def get_num_edges(file):
        with open(file, encoding="utf8") as f:
            num_edges, _, _ = f.readline().split()
            num_edges = int(num_edges)
            return num_edges

    cmp_f = lambda x, y: x - y
    cmp_lst = []
    for design in design_lst:
        hg_file = os.path.join("res", "ispd2005", design, design + ".hg")
        vir_file = os.path.join("res", "ispd2005", design, design + ".vir")
        hg_num_edges = get_num_edges(hg_file)
        vir_num_edges = get_num_edges(vir_file)
        cmp_lst.append(cmp_f(vir_num_edges, hg_num_edges))
    fig, ax = plt.subplots()
    ax.scatter(design_lst, cmp_lst)
    ax.axhline(y=0, c="r", ls="-")
    ax.set_ylabel(
        # r"$\frac{edge_{vir}-edge_{hg}}{edge_{hg}}$",
        r"$edge_{vir}-edge_{hg}$",
        fontdict={"size": 12},
        rotation=0,
        loc="top",
        labelpad=-80,
    )
    fig.savefig("res/ispd2005/conclude.num_edge.png", dpi=300)
    plt.close()


def conclude_table(config, conclude, file_name):
    design_list = config["design"]
    k_list = config["KaHyPar"]["k"]
    design_list.sort()
    k_list.sort()
    df_value = pd.DataFrame(index=k_list, columns=design_list)
    df_hpwl = pd.DataFrame(index=k_list, columns=design_list)
    for instance, conclude_info in conclude.items():
        design, k = instance.split(".")
        k = int(k)
        df_value.loc[k, design] = conclude_info["value"]
        df_hpwl.loc[k, design] = conclude_info["hpwl"]
    df_value.to_csv(file_name + ".value.csv")
    df_hpwl.to_csv(file_name + ".hpwl.csv")
    return df_value, df_hpwl


def conclude_all():
    base_file = os.path.join("res", "ispd2005", "conclude.hg.origin.json")
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = jstyleson.load(f)
    transforms = ["shrink1", "shrink", "base", "enlarge", "enlarge10000"]
    conclude_info = {}
    for trans in transforms:
        conclue_file = os.path.join("res", "ispd2005", f"conclude.vir.{trans}.json")
        with open(conclue_file, "r", encoding="utf8") as f:
            conclude_info[trans] = jstyleson.load(f)
    benchmarks = [k for k in conclude_info[transforms[0]].keys() if "500" in k]  # 如果全画，图太多了，只画k=500的

    os.system("mkdir -p res/ispd2005/conclude")
    for bm in benchmarks:
        res_pic_name = os.path.join("res", "ispd2005", "conclude", f"conclude.{bm}")
        value_list, hpwl_list, ncut_list = [], [], []
        for trans in transforms:
            value_list.append(
                (conclude_info[trans][bm]["value"] - baseline[bm]["value"]) / baseline[bm]["value"]
            )
            hpwl_list.append(
                -(conclude_info[trans][bm]["hpwl"] - baseline[bm]["hpwl"]) / baseline[bm]["hpwl"]
            )
            ncut_list.append((conclude_info[trans][bm]["ncut"] - baseline[bm]["ncut"]) / baseline[bm]["ncut"])
        fig, ax = plt.subplots()
        # plot value
        ax.set_xlabel("trans")
        ax.set_ylabel(
            r"$\frac{value_{mod}-value_{ori}}{value_{ori}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax.scatter(transforms, value_list, c="b")
        fig.savefig(res_pic_name + ".value.png", dpi=300)
        plt.close(fig)

        # plot hpwl
        fig2, ax2 = plt.subplots()
        ax2.set_ylabel(
            r"$-\frac{hpwl_{mod}-hpwl_{ori}}{hpwl_{ori}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax2.scatter(transforms, hpwl_list, c="r")
        fig2.savefig(res_pic_name + ".hpwl.png", dpi=300)
        plt.close(fig2)

        # plot ncut
        fig3, ax3 = plt.subplots()
        ax3.set_ylabel(
            r"$\frac{ncut_{mod}-ncut_{ori}}{ncut_{ori}}$",
            fontdict={"size": 16},
            rotation=0,
            loc="top",
            labelpad=-90,
        )
        ax3.scatter(transforms, ncut_list, c="c")
        fig3.savefig(res_pic_name + ".ncut.png", dpi=300)
        plt.close(fig3)


def conclude_hpwl():
    base_file = os.path.join("res", "ispd2005", "conclude.hg.origin.json")
    with open(base_file, "r", encoding="utf-8") as f:
        baseline = jstyleson.load(f)
    mw_file = os.path.join("res", "ispd2005", "conclude.hg.hpwl.json")
    with open(mw_file, "r", encoding="utf-8") as f:
        mw = jstyleson.load(f)

    benchmarks = list(mw.keys())
    os.system("mkdir -p res/ispd2005/conclude")
    value_list, hpwl_list, ncut_list = [], [], []
    for bm in benchmarks:
        value_list.append((mw[bm]["value"] - baseline[bm]["value"]) / baseline[bm]["value"])
        hpwl_list.append(-(mw[bm]["hpwl"] - baseline[bm]["hpwl"]) / baseline[bm]["hpwl"])
        ncut_list.append((mw[bm]["ncut"] - baseline[bm]["ncut"]) / baseline[bm]["ncut"])

    res_pic_name = os.path.join("res", "ispd2005", "conclude", f"conclude.hpwl")
    fig, ax = plt.subplots(figsize=(8,11))
    # plot value
    ax.set_xlabel("trans")
    ax.set_ylabel(
        r"$\frac{value_{mod}-value_{ori}}{value_{ori}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax.axhline(y=0, c="r", ls="-")
    ax.scatter(benchmarks, value_list, c="b")
    plt.xticks(rotation=-90)
    fig.savefig(res_pic_name + ".value.pdf", dpi=300)
    plt.close(fig)

    # plot hpwl
    fig2, ax2 = plt.subplots(figsize=(8,11))
    ax2.set_ylabel(
        r"$-\frac{hpwl_{mod}-hpwl_{ori}}{hpwl_{ori}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax2.axhline(y=0, c="r", ls="-")
    ax2.scatter(benchmarks, hpwl_list, c="r")
    plt.xticks(rotation=-90)
    fig2.savefig(res_pic_name + ".hpwl.pdf", dpi=300)
    plt.close(fig2)

    # plot ncut
    fig3, ax3 = plt.subplots(figsize=(8,11))
    ax3.set_ylabel(
        r"$\frac{ncut_{mod}-ncut_{ori}}{ncut_{ori}}$",
        fontdict={"size": 16},
        rotation=0,
        loc="top",
        labelpad=-90,
    )
    ax3.axhline(y=0, c="r", ls="-")
    ax3.scatter(benchmarks, ncut_list, c="c")
    plt.xticks(rotation=-90)
    fig3.savefig(res_pic_name + ".ncut.pdf", dpi=300)
    plt.close(fig3)


if __name__ == "__main__":
    # type = "k"  # 'k' or 'g', k 表示按照切分数量优先, g 表示按照图有限
    # vir_conclude_file = "res/ispd2005/conclude.hg.hpwl.json"
    # hg_conclude_file = "res/ispd2005/conclude.hg.origin.json"
    # config_file = os.path.join("par_config", "ispd2005", "config.json")
    # with open(vir_conclude_file, "r", encoding="utf-8") as f:
    #     vir_conclude = jstyleson.load(f)
    # with open(hg_conclude_file, "r", encoding="utf-8") as f:
    #     hg_conclude = jstyleson.load(f)
    # with open(config_file, encoding="utf-8") as f:
    #     config = jstyleson.load(f)
    # idx_list, k_list, val_cmp_list, hpwl_cmp_list = compare(vir_conclude, hg_conclude, type)
    # plot_cmp(val_cmp_list, idx_list, type, "value")
    # plot_cmp(hpwl_cmp_list, idx_list, type, "hpwl")
    # conclude_table(config, hg_conclude, "res/ispd2005/conclude.hg.origin")
    # conclude_table(config, vir_conclude, "res/ispd2005/conclude.hg.hpwl")

    # config_file = os.path.join("par_config", "ispd2005", "config.json")
    # with open(config_file, encoding="utf-8") as f:
    #     config = jstyleson.load(f)
    # plot_improve(config["design"])
    # test_HPWL()

    # conclude_all()
    conclude_hpwl()
