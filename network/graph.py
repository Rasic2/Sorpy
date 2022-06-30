from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from common.io_file import POSCAR
from common.logger import root_dir

fname = Path(f'{root_dir}/train_set/input/POSCAR_1-1')
s1 = POSCAR(fname=fname).to_structure(style="Slab")
s1.find_neighbour_table(neighbour_num=12)
atom_feature = np.array([[atom.element.group, atom.element.period] for atom in s1.atoms])
adj_matrix = s1.NNT.__index
dist_feature = s1.NNT.dist
dist3d_feature = s1.NNT.dist3d

edges = []
for key, value in s1.NNT.items():
    for v in value:
        edges.append([key.element.formula+str(key.order), v[0].element.formula+str(v[0].order)])

G = nx.Graph()
G.add_nodes_from([atom.element.formula+str(atom.order) for atom in s1.atoms])
G.add_edges_from(edges)
nodelist = G.nodes
node_color=[atom.element.__elements[f'Element {atom.element.formula}']['color'] for atom in s1.atoms]

nx.draw_networkx(G, nodelist=G.nodes,node_color=node_color,node_size=500,font_size=10)
plt.show()