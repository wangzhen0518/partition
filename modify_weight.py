import pickle as pk
import numpy as np
import os


def rebuild():
    design_list = [
        "adaptec1",
        "adaptec2",
        "adaptec3",
        "adaptec4",
        "bigblue1",
        "bigblue2",
    ]
    b_pth = "./res/ispd2005"
    for design in design_list:
        print(design)
        vir_edge_file = os.path.join(b_pth, design, "vir_edge.bin")
        with open(vir_edge_file, "rb") as f:
            vir_edge = pk.load(f)
        for i, (w, h, t) in enumerate(vir_edge):
            vir_edge[i] = [int(1), h, t]
        vir_edge_file = os.path.join(b_pth, design, "vir_edge.shrink1.bin")
        with open(vir_edge_file, "wb") as f:
            pk.dump(vir_edge, f)


def rebuild_vir_file():
    design_list = [
        "adaptec1",
        # "adaptec2",
        # "adaptec3",
        # "adaptec4",
        # "bigblue1",
        # "bigblue2",
    ]
    b_pth = "./res/ispd2005"
    for design in design_list:
        print(design)
        vir_edge_file = os.path.join(b_pth, design, f"{design}.vir")
        with open(vir_edge_file, "r") as f:
            vir_edge = f.readlines()
        vir_edge_new = []
        for i, e in enumerate(vir_edge):
            e.replace("\n", "")
            if i != 0:
                if e != "\n":
                    w = int(eval(e.split()[0]))
                    if w != 1:
                        w, h, t = [eval(i) for i in e.split()]
                        vir_edge_new.append(f"{int(w/50*20)} {h} {t}\n")
                    else:
                        vir_edge_new.append(e)
            else:
                vir_edge_new.append(e)
        s = "".join(vir_edge_new)
        vir_edge_file = os.path.join(b_pth, design, f"{design}.vir")
        with open(vir_edge_file, "w") as f:
            f.write(s)


if __name__ == "__main__":
    rebuild()
