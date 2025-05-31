import collections
from collections import OrderedDict
from typing import Dict, Any


class CacheManager:
    def __init__(self, cache_size_per_node: int, num_edge_nodes: int, replacement_policy: str,
                 cache_sharing_policy: str, public_cache_fraction: float = 0.0):
        """
        Initializes the CacheManager with specified parameters.

        Args:
            cache_size_per_node (int): The maximum number of functions an individual edge node can cache.
            num_edge_nodes (int): The total number of edge nodes in the simulation.
            replacement_policy (str): The cache replacement policy (e.g., "LRU").
            cache_sharing_policy (str): The cache sharing policy ("full_private", "full_public", "partial_public_private").
            public_cache_fraction (float): Fraction of total capacity dedicated to public cache in "partial_public_private" policy.
        """
        self.cache_size_per_node = cache_size_per_node
        self.num_edge_nodes = num_edge_nodes
        self.replacement_policy = replacement_policy  # Currently only LRU is implemented/simulated
        self.cache_sharing_policy = cache_sharing_policy
        self.public_cache_fraction = public_cache_fraction

        self.private_caches: Dict[int, OrderedDict] = {}  # For "full_private" and "partial_public_private"
        self.public_cache: OrderedDict = OrderedDict()  # For "full_public" and "partial_public_private"
        self.public_cache_size: int = 0  # Calculated size for the public cache
        self.private_cache_size_for_partial: int = 0  # Calculated private size per node for "partial_public_private"

        self._initialize_caches()

    def _initialize_caches(self):
        """Initializes cache structures based on the chosen sharing policy."""
        if self.cache_sharing_policy == "full_private":
            for i in range(self.num_edge_nodes):
                self.private_caches[i] = OrderedDict()
            print(
                f"CacheManager: Initialized with 'full_private' policy. Each node has a private cache of size {self.cache_size_per_node}.")

        elif self.cache_sharing_policy == "full_public":
            # In full_public, all nodes share one large cache.
            # The size of this public cache is total_nodes * cache_size_per_node, conceptually.
            self.public_cache_size = self.cache_size_per_node * self.num_edge_nodes
            self.public_cache = OrderedDict()  # Re-initialize to ensure it's empty
            print(
                f"CacheManager: Initialized with 'full_public' policy. Global public cache size: {self.public_cache_size}.")

        elif self.cache_sharing_policy == "partial_public_private":
            # Calculate sizes for public and private portions
            total_logical_cache_capacity = self.cache_size_per_node * self.num_edge_nodes

            # Global Public Cache: A portion of the total logical capacity is pooled into a single public cache.
            self.public_cache_size = max(1, int(total_logical_cache_capacity * self.public_cache_fraction))
            self.public_cache = OrderedDict()  # Re-initialize to ensure it's empty

            # Private Caches per Node: Each node has a remaining private capacity.
            # Here, we model it as a fixed size per node for its private portion.
            self.private_cache_size_for_partial = max(1, self.cache_size_per_node - int(
                self.cache_size_per_node * self.public_cache_fraction))

            # Initialize private caches for each node
            for i in range(self.num_edge_nodes):
                self.private_caches[i] = OrderedDict()

            print(f"CacheManager: Initialized with 'partial_public_private' policy.")
            print(f"  Global public cache size: {self.public_cache_size}")
            print(f"  Each node's private cache size: {self.private_cache_size_for_partial}")

        else:
            raise ValueError(f"Unsupported cache_sharing_policy: {self.cache_sharing_policy}")

    def is_cached(self, node_id: int, function_id: str) -> bool:
        """
        Checks if a function is cached on a given node (or globally, depending on policy).

        Args:
            node_id (int): The ID of the edge node.
            function_id (str): The ID of the function to check.

        Returns:
            bool: True if the function is cached, False otherwise.
        """
        if self.cache_sharing_policy == "full_private":
            return function_id in self.private_caches[node_id]

        elif self.cache_sharing_policy == "full_public":
            return function_id in self.public_cache

        elif self.cache_sharing_policy == "partial_public_private":
            # Function is considered cached if it's in the node's private cache OR in the global public cache.
            return (function_id in self.private_caches[node_id] or
                    function_id in self.public_cache)

        return False  # Should not be reached with valid policies

    def access_function(self, node_id: int, function_id: str):
        """
        Simulates accessing a function, updating cache status (LRU).
        This method should be called when a task requests a function.

        Args:
            node_id (int): The ID of the edge node accessing the function.
            function_id (str): The ID of the function being accessed.
        """
        if self.cache_sharing_policy == "full_private":
            cache = self.private_caches[node_id]
            self._update_lru_cache(cache, function_id, self.cache_size_per_node)

        elif self.cache_sharing_policy == "full_public":
            # All accesses update the single public cache
            self._update_lru_cache(self.public_cache, function_id, self.public_cache_size)

        elif self.cache_sharing_policy == "partial_public_private":
            private_cache = self.private_caches[node_id]

            # 1. Check if already in the node's private cache
            if function_id in private_cache:
                self._update_lru_cache(private_cache, function_id, self.private_cache_size_for_partial)

            # 2. If not in private, but in public cache:
            elif function_id in self.public_cache:
                # Update public cache LRU, indicating it was used (fetched from public pool)
                self.public_cache.move_to_end(function_id)
                # Also try to load it into the private cache if there's space, or if LRU allows an eviction
                self._update_lru_cache(private_cache, function_id, self.private_cache_size_for_partial)

            # 3. If not in private AND not in public: It's a completely cold function. Load into private.
            else:
                self._update_lru_cache(private_cache, function_id, self.private_cache_size_for_partial)

    def _update_lru_cache(self, cache: OrderedDict, key: str, max_size: int):
        """
        Helper method to perform LRU update/insertion on a given OrderedDict cache.

        Args:
            cache (OrderedDict): The cache dictionary to update.
            key (str): The key (function_id) to access.
            max_size (int): The maximum size of this specific cache.
        """
        if key in cache:
            cache.move_to_end(key)  # Move to end (most recently used)
        else:
            if len(cache) >= max_size:
                cache.popitem(last=False)  # Evict the least recently used item (first item)
            cache[key] = True  # Add the new item

    def predict_and_load(self, node_id: int, functions_to_load: list):
        """
        Simulates pre-loading functions into cache based on a prediction model.
        This is a placeholder for future, more sophisticated prediction logic (Phase 3 Goal 3).

        Args:
            node_id (int): The ID of the edge node for which to pre-load functions.
            functions_to_load (list): A list of function_ids to attempt to load.
        """
        print(f"CacheManager: Attempting predictive load for node {node_id} with functions: {functions_to_load}")
        if self.cache_sharing_policy == "full_private":
            cache = self.private_caches[node_id]
            for func_id in functions_to_load:
                if func_id not in cache:
                    self._update_lru_cache(cache, func_id, self.cache_size_per_node)

        elif self.cache_sharing_policy == "full_public":
            for func_id in functions_to_load:
                if func_id not in self.public_cache:
                    self._update_lru_cache(self.public_cache, func_id, self.public_cache_size)

        elif self.cache_sharing_policy == "partial_public_private":
            private_cache = self.private_caches[node_id]
            for func_id in functions_to_load:
                # Try to load into private portion first if space.
                if func_id not in private_cache and len(private_cache) < self.private_cache_size_for_partial:
                    private_cache[func_id] = True
                # If private is full or function already there, and not in public, try public pool.
                elif func_id not in self.public_cache:
                    self._update_lru_cache(self.public_cache, func_id, self.public_cache_size)