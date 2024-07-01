import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

def buildGraph():
    # Create an empty directed graph
    G = nx.DiGraph()

    # Read the dataset and add edges to the graph
    with open('./dataset/training/Centrality.csv', 'r') as file:
        next(file)  # Skip the header
        for line in file:
            author, mention = line.strip().split(',')
            G.add_edge(author, mention)

    G = G.to_undirected()
    return G

def netShield(A, k, seed_indices):
    # Compute the largest eigenvalue and corresponding eigenvector of the adjacency matrix
    A = A.astype(float)
    eigenvalue, eigenvector = eigsh(A, k=1, which='LM')
    eigenvalue = eigenvalue[0]
    eigenvector = eigenvector[:, 0]

    n = A.shape[0]
    selected_nodes = []

    # Compute initial shield values for each node
    v = 2 * eigenvalue * (eigenvector ** 2) - np.diag(A) * (eigenvector ** 2)

    for _ in range(k):
        scores = v.copy()
        # Adjust scores for nodes already selected or in seed_nodes
        for i in selected_nodes + seed_indices:
            scores[i] = -np.inf

        # Adjust scores for nodes already selected
        for i in selected_nodes:
            scores -= 2 * A[:, i] * eigenvector[i] * eigenvector

        # Select the node with the maximum shield value
        max_score_node = np.argmax(scores)
        selected_nodes.append(max_score_node)

        # Update v to reflect the removal of the selected node
        v -= 2 * A[:, max_score_node] * eigenvector[max_score_node] * eigenvector

    return selected_nodes

def apply_netShield_on_reachable_subgraph(G, seeds, k):
    # Get the subgraph reachable from seeds
    reachable_nodes = set()
    for seed in seeds:
        reachable_nodes.update(nx.single_source_shortest_path_length(G, seed).keys())

    subgraph = G.subgraph(reachable_nodes)

    # Create a mapping from node names to indices
    nodelist = list(subgraph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodelist)}
    index_to_node = {i: node for i, node in enumerate(nodelist)}

    # Convert seed names to indices
    seed_indices = [node_to_index[seed] for seed in seeds]

    # Get the adjacency matrix of the subgraph
    A = nx.adjacency_matrix(subgraph, nodelist=nodelist).toarray()

    # Apply NetShield to the subgraph
    S = netShield(A, k, seed_indices)

    # Map the result back to the original graph's node names
    influential_nodes_netshield = [index_to_node[i] for i in S]
    return influential_nodes_netshield


class SimpleSimulator:

    def __init__(self, G, seeds):
        self.G = G
        self.seeds = seeds

    def simulate(self, blocked):
        # Perform the simulation without blocking first
        activated_without_blocking = self.simulate_spread(self.G, self.seeds)

        # Perform the simulation with blocking
        G_blocked = self.G.copy()
        G_blocked.remove_nodes_from(blocked)
        # Update seeds after removing blocked nodes
        updated_seeds = [seed for seed in self.seeds if seed not in blocked]
        activated_with_blocking = self.simulate_spread(G_blocked, updated_seeds)

        # Calculate saved nodes
        saved_nodes = set(activated_without_blocking) - set(activated_with_blocking)
        return len(saved_nodes), len(activated_without_blocking), len(activated_with_blocking)
    
    def simulate_spread(self, G, seeds):
        active = set(seeds)
        front_nodes = seeds

        while front_nodes:
            new_front_nodes = []
            for node in front_nodes:
                # Use successors for directed graphs, neighbors for undirected graphs
                neighbors = G.neighbors(node)

                for neighbor in neighbors:
                    if neighbor not in active and np.random.rand() <= G[node][neighbor].get('weight', 1):
                        active.add(neighbor)
                        new_front_nodes.append(neighbor)
            front_nodes = new_front_nodes

        return active

graph = buildGraph()
simulator = SimpleSimulator(graph, ['LoneStarSUVLimo'])
print(apply_netShield_on_reachable_subgraph(graph, ['LoneStarSUVLimo'], 2))
result = simulator.simulate(apply_netShield_on_reachable_subgraph(graph, ['LoneStarSUVLimo'], 2))
print(result)
            # st.write(result)
            # st.write("Done!")
            # st.markdown(f'<div class="custom-container"><div class="stAlert">✅ Managed to save 5 users.</div></div>', unsafe_allow_html=True)    
