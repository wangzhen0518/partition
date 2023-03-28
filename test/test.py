import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    g=nx.Graph()
    g.add_node(1)
    g.add_node('A')
    g.add_nodes_from([2,3])
    g.add_edges_from([(1,2),(1,3),(2,4),(2,5),(3,6),(4,8),(5,8),(3,7)])
    h=nx.path_graph(10)
    g.add_nodes_from(h)
    g.add_node(h)
    g.add_node(1,1)
    g.add_edge('x','y')
    g.add_weighted_edges_from(['x','y',1.])
    lst=[[('a','b',5.),('b','c',3.),('a','c',1.)]]
    g.add_weighted_edges_from([(lst)])
    
    h