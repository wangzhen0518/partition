import subprocess
import os
import jstyleson
import numpy as np
import matplotlib.pyplot as plt

from utils import analysis_stats
from manager import generate_par, eval_par

if __name__ == "__main__":
    hg_file = "res/ispd2005/adaptec1/adaptec1.hg"
    pl_file = "res/ispd2005/adaptec1/adaptec1.gp.pl"
    res_json = "res/ispd2005/adaptec1/stable.ln.json"
    res_fig = "res/ispd2005/adaptec1/stable.ln.png"
    k = 2
    N = 10
    res_dict = dict()
    for i in range(N):
        # stat, res = subprocess.getstatusoutput(f"shmetis {hg_file} {k} 2")
        stat, res = 0, ""
        target_file = hg_file + f".{k}.{i}"
        if stat == 0:
            # os.system(f"mv {hg_file}.part.{k} {target_file}")
            par_dict = generate_par(target_file, pl_file)
            val, _ = eval_par(par_dict)
            key = os.path.basename(target_file)
            res_dict[key] = np.log(val)
            print(key, val)
        else:
            print(f"Error {res}")
    with open(res_json, "w", encoding="utf8") as f:
        jstyleson.dump(res_dict, f, sort_keys=True, indent=4)
    val_lst = np.array(list(res_dict.values()))
    val_aver = np.average(val_lst)
    var = np.var(val_lst)
    val_lst = (val_lst - val_aver) / val_aver
    plt.scatter(range(len(val_lst)), val_lst)
    plt.axhline(y=0, c="r", ls="-")
    plt.text(0, 0.05, f"$\ln(var)$={np.log(var):.2f}")
    plt.savefig(res_fig, dpi=300)
    plt.close()
