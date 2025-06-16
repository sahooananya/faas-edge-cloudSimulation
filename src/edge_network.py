# src/edge_network.py

import networkx as nx
from .models import EdgeNode


class EdgeNetwork:
    def __init__(self, num_edge_nodes: int, base_latency: float,
                 adjacency_matrix=None):  # base_latency now represents edge-to-edge
        self.num_edge_nodes = num_edge_nodes
        self.edge_nodes = [EdgeNode(i) for i in range(num_edge_nodes)]
        self.base_latency = base_latency  # This is the EDGE_TO_EDGE_LATENCY

        self.graph = nx.Graph()
        self._build_network_graph(num_edge_nodes, adjacency_matrix)

    def _build_network_graph(self, num_edge_nodes, adjacency_matrix):
        for i in range(num_edge_nodes):
            self.graph.add_node(i, node_obj=self.edge_nodes[i])

        if adjacency_matrix:
            for i in range(num_edge_nodes):
                for j in range(i + 1, num_edge_nodes):
                    if adjacency_matrix[i][j] == 1:
                        # Use self.base_latency (which is EDGE_TO_EDGE_LATENCY) for edge weights
                        self.graph.add_edge(i, j, latency=self.base_latency)
        else:
            # Fallback to fully connected with default latency if no matrix
            for i in range(num_edge_nodes):
                for j in range(i + 1, num_edge_nodes):
                    self.graph.add_edge(i, j, latency=self.base_latency)

        print(
            f"Edge Network Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def get_neighbors(self, node_id: int):
        neighbors = []
        if node_id in self.graph:
            for neighbor_id in self.graph.neighbors(node_id):
                neighbors.append(self.graph.nodes[neighbor_id]['node_obj'])
        return neighbors

    def get_all_edge_nodes(self):
        return self.edge_nodes

    def get_edge_node_by_id(self, node_id: int):
        """Returns the EdgeNode object for a given ID."""
        if 0 <= node_id < self.num_edge_nodes:
            return self.edge_nodes[node_id]
        return None