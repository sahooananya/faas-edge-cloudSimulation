# src/cache_manager.py

import collections

class CacheManager:
    def __init__(self, cache_size_per_node, num_edge_nodes, policy="LRU", sharing_policy="full_private"):
        self.cache_size_per_node = cache_size_per_node
        self.num_edge_nodes = num_edge_nodes
        self.policy = policy # Only LRU is implemented for Phase 1
        self.sharing_policy = sharing_policy # Only "full_private" is relevant for Phase 1

        self.edge_caches = {} # {edge_node_id: OrderedDict (function_id: True)}
        for i in range(num_edge_nodes):
            self.edge_caches[i] = collections.OrderedDict()

        # For Phase 1, other sharing policies are not active.
        # self.public_cache = collections.OrderedDict() # For "full_public" in Phase 2

    def is_cached(self, node_id, function_id):
        """Checks if a function is in the cache of a specific edge node."""
        if self.sharing_policy == "full_private":
            return function_id in self.edge_caches[node_id]
        # Other sharing policies will be handled in Phase 2

        return False # Default if no policy matches

    def access_function(self, node_id, function_id):
        """Simulates accessing a function, updating cache and potentially evicting."""
        if self.sharing_policy == "full_private":
            self._access_private_cache(node_id, function_id)
        # Other sharing policies will be handled in Phase 2

    def _access_private_cache(self, node_id, function_id):
        """LRU logic for a single private cache."""
        if function_id in self.edge_caches[node_id]:
            # Function is in cache, move to end (most recently used)
            self.edge_caches[node_id].move_to_end(function_id)
        else:
            # Function not in cache, add it. Evict LRU if cache is full.
            if len(self.edge_caches[node_id]) >= self.cache_size_per_node:
                self.edge_caches[node_id].popitem(last=False) # Remove oldest item
            self.edge_caches[node_id][function_id] = True # Add new item

    # Predictive cache loading will be added in Phase 2
    def predict_and_load(self, node_id, function_ids_to_load):
        pass # Placeholder for Phase 2