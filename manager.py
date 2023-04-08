import numpy as np

from utils import load_position, load_par, dict_append, del_ext_name, generate_benchmark_dict

# """
# partition的数据结构为
# {
#     ...,
#     k: [(id, x, y), ...],
#     ...
# }
# """
"""
partition的数据结构为
{
    ...,
    k: [n, array(id), array(xi), array(yi)],
    ...
}

n: 第k个类中点的数量
"""

def generate_par(par_file, pl_file):
    print(par_file, pl_file)
    par = load_par(par_file)
    pos_x, pos_y = load_position(pl_file)
    pos_x, pos_y = np.array(pos_x), np.array(pos_y)

    n = len(par)
    par_dict = dict()
    # 先为每个类构建array(id)
    for _id, k in enumerate(par):
        dict_append(par_dict, k, _id)
    # 为每一个类构建[n, array(id), array(x), array(y)]
    for k, id_list in par_dict.items():
        id_list = np.array(id_list)
        x = pos_x[id_list]
        y = pos_y[id_list]
        nk = len(id_list)
        par_dict[k] = [nk, id_list, x, y]
    return par_dict

def eval(par: dict):
    # TODO 改成 tensor 版本，加速
    # 先将partition转换成numpy.array
    # val = 1/N sum_k sum_i [(xi-x_bar)^2 + (yi-y_bar)^2]
    N = 0
    val = 0
    val_list = []
    for k, (n, _id, x, y) in par.items():
        N += n
        val_tmp = x.var() + y.var()
        # val += val_tmp * n
        val += val_tmp
        val_list.append(val_tmp)
    val /= len(par)
    # val /= N
    return val, val_list


def num_par(par: dict):
    """
    看各个切分的数量是否均衡
    """
    num = [n for n, _, _, _ in par.values()]
    var = np.var(num)
    # print(var)
    print(num)




if __name__ == "__main__":
    import glob, os

    bench_dict = generate_benchmark_dict("ispd2005", "shmetis_method")

    # par_pth = "res/ispd2005/shmetis_method/par"
    # pl_pth = "res/ispd2005/pl"
    # par_list = glob.glob(os.path.join(par_pth, "*"))
    # pl_list = glob.glob(os.path.join(pl_pth, "*"))
    # par_list.sort()
    # pl_list.sort()

    # for par_file, pl_file in zip(par_list, pl_list):
    #     par = generate_par(par_file, pl_file)
    #     var = eval(par)
    #     print(f"{par_file}: {var:.4f}")

    for design, d in bench_dict.items():
        pl_file = d["pl"]
        par_list = d["par"]
        for par_file in par_list:
            par = generate_par(par_file, pl_file)
            var, var_list = eval(par)
            # print(f"{pl_file} {par_file}: {var:.4f}, {var_list}")
            print(f"{pl_file} {par_file}: {var:.4f}")
