import networkx as nx
import sys
sys.path.append('/Robust-Branching-master/')
import matplotlib.pyplot as plt
from data_augmentation.augmentation_operator import *


"""
生成pos,用来配置layout,其格式为{0: [-0.95, -0.8],1: [-0.95, -0.48]}
"""
def gen_pos(node_type_num, node_num_array):
    step=round(1.6/(node_type_num-1), 2)
    xs=[]
    for i in range(node_type_num):
        xs.append(round(-0.95+i*step,2))
    
    ys=[]
    for node_num in node_num_array:
        cstep=round(1.6/(node_num-1),5)
        result=[]
        for j in range(node_num):
            result.append(round(-0.8+j*cstep,5))
        ys.append(result) 
        
    #construct pos
    pos=[]
    for aa,bb in zip(xs, ys):
        for b in bb:
            pos.append([aa,b])
    
    final_pos={}
    for i,e in enumerate(pos):
        final_pos[i]=e
    return final_pos

[obj, var2idx, constraint_features, cons_indices, cons_features] = read_instance('/data/hyliu/ML4CO/learn2branch-master/data/instances/setcover/transfer_1000r_1000c_0.01d/instance_2.lp')

pos=gen_pos(2,[len(var2idx), len(cons_indices)])

## 创建图对象，并添加节点，添加边 
GG2=nx.DiGraph()

# add nodes
GG2.add_nodes_from(list(range(len(var2idx)+len(cons_indices))))  
# add edges
u = []
v = []
for i in range(len(cons_indices)):
    for j in cons_indices[i]:
        u.append(1000-j)
        v.append(i)
res2=[]
res3=[]
for i,j in zip(u,v):
    res2.append((i,j+len(var2idx)))
GG2.add_edges_from(res2)


# nodes, 把节点分为三个组(相当于三种类型)，每组独立编号与染色
plt.figure(figsize=(50,650))
nx.draw_networkx_nodes(GG2, pos, nodelist=list(range(len(var2idx))), node_color="red",label="A")
nx.draw_networkx_nodes(GG2, pos, nodelist=list(range(len(var2idx),len(var2idx)+len(cons_indices))), node_color="green",label="B")


#edges，把边分为两个组(相当于两种边类型)，每组独立设置样式
nx.draw_networkx_edges(GG2, pos, edgelist=res2, width=1)

# node labels，每类节点从0开始编号
labels = dict(zip(range(len(var2idx)+len(cons_indices)),list(range(len(var2idx)))+list(range(len(cons_indices)))))
nx.draw_networkx_labels(GG2, pos, labels, font_size=0.5, font_color="whitesmoke")


# legend1：渲染节点类型，labelspacing设置节点间距离(垂直方向)、borderpad设置节点与边界间距离(垂直方向)
l1=plt.legend(bbox_to_anchor=(1,0.85),labelspacing=1,borderpad=0.7)
plt.gca().add_artist(l1)       # 这条语句可使plt添加多个legend

# legend2：渲染边类型
from matplotlib.lines import Line2D
handles=[Line2D([],[],color="black",label="A->B",linewidth=0.1)]
plt.legend(handles=handles, bbox_to_anchor=(1,0.6))

# dpi设置清晰度、bbox_inches='tight'保证图片能被完整保存
plt.savefig("/home/hyliu/ML4CO/Robust-Branching-master/data_augmentation/instance2_2",dpi=100,bbox_inches = 'tight')