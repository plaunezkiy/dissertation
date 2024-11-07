import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes with data associated with them
G.add_node(1, label='Node A')
G.add_node(2, label='Node B')
G.add_node(3, label='Node C')

# Add edges with data associated with them
G.add_edge(1, 2, weight=4, label='Edge 1->2')
G.add_edge(2, 3, weight=5, label='Edge 2->3')
G.add_edge(3, 1, weight=6, label='Edge 3->1')

# Position the nodes using a layout
pos = nx.spring_layout(G)

# Draw the nodes with labels
node_labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_nodes(G, pos, node_size=700)
nx.draw_networkx_labels(G, pos, labels=node_labels)

# Draw the edges with labels
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Remove axis
plt.axis('off')

# Display the graph
plt.show()
