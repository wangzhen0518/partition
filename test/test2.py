import matplotlib.pyplot as plt
import networkx as nx

g = nx.Graph()
h = nx.path_graph(10)
g.add_nodes_from(h)
nx.draw(g, with_labels=True)
plt.show()
