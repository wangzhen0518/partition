import os
import jstyleson
from multiprocessing import Pool

if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from main import load_design
    from hypergraph import DiHypergraph
    import hyperdata

    benchmark = "ispd2005"
    b_pth = os.path.join("..", "res", benchmark)
    config_file = os.path.join("..", "par_config", benchmark, "config.json")
    with open(config_file, encoding="utf8") as f:
        config = jstyleson.load(f)

    # load hypergraph
    num_thread = 8
    pool = Pool(num_thread)
    hg_list: list[DiHypergraph] = []
    task_list = []
    design_list = config["design"]
    for design in design_list:
        # hg = load_design(benchmark, design, b_pth, use_vir)
        # hg_lst.append(hg)
        task_list.append(pool.apply_async(load_design, args=(benchmark, design, b_pth, False)))
    pool.close()
    pool.join()
    # hg_list = []
    for task in task_list:
        hg = task.get()
        hg_list.append(hg)

    par_file_list = []
    for hg in hg_list:
        par_file = os.path.join(hg.design_pth,'shmetis', hg.design + ".hg.2.5")
        par_file_list.append(par_file)
    hg_set = hyperdata.HypergraphDataset(benchmark, hg_list, par_file_list, verbose=True)
