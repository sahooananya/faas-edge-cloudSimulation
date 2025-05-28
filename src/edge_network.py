# src/edge_network.py

from .models import EdgeNode

class EdgeNetwork:
    def __init__(self, num_edge_nodes, base_latency=0.01):
        self.edge_nodes = [EdgeNode(i) for i in range(num_edge_nodes)]
        self.base_latency = base_latency # Base communication latency (not used for scheduling in Phase 1)
        # Adjacency graph (networkx) will be added in Phase 2

    def get_edge_node_by_id(self, node_id):
        if 0 <= node_id < len(self.edge_nodes):
            return self.edge_nodes[node_id]
        return None

    def get_all_edge_nodes(self):
        return self.edge_nodes

    # Methods for adjacency and routing will be added in Phase 2