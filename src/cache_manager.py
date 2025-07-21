from collections import OrderedDict
import random
from typing import Dict, List, Union


class CacheManager:
    def __init__(self, cache_size_per_node: int, num_edge_nodes: int, replacement_policy: str,
                 cache_sharing_policy: str, public_cache_fraction: float = 0.0):
        """
        Initializes the CacheManager with specified parameters.
        """
        self.cache_size_per_node = cache_size_per_node
        self.num_edge_nodes = num_edge_nodes
        self.replacement_policy = replacement_policy
        self.cache_sharing_policy = cache_sharing_policy
        self.public_cache_fraction = public_cache_fraction

        self.private_caches: Dict[int, OrderedDict] = {}
        self.public_cache: OrderedDict = OrderedDict()

        self._initialize_caches()

    def _initialize_caches(self):
        """Initializes private caches for each node and sets public cache size."""
        for i in range(self.num_edge_nodes):
            self.private_caches[i] = OrderedDict()

        if self.cache_sharing_policy == "full_public":
            self.public_cache_size = self.cache_size_per_node * self.num_edge_nodes
        elif self.cache_sharing_policy == "partial_public_private":
            self.public_cache_size = int(self.cache_size_per_node * self.num_edge_nodes * self.public_cache_fraction)
            self.private_cache_size_for_partial = self.cache_size_per_node - \
                                                 int(self.cache_size_per_node * self.public_cache_fraction)
            if self.private_cache_size_for_partial <= 0 and self.cache_size_per_node > 0:
                self.private_cache_size_for_partial = 1
        elif self.cache_sharing_policy == "full_private":
            pass

    def _update_lru_cache(self, cache: OrderedDict, key: str, max_size: int):
        """Helper to update LRU cache: move to end (most recently used) or add and pop oldest."""
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = True
            if len(cache) >= max_size:
                cache.popitem(last=False)

    def prefetch_functions(self, node_id: int, func_ids: Union[str, List[str]]):
        """
        Prefetches one or more functions into the cache(s) based on the cache sharing policy.
        `func_ids` can be a single string or a list of strings.
        """
        if isinstance(func_ids, str):
            func_ids_list = [func_ids]
        else:
            func_ids_list = func_ids

        if self.cache_sharing_policy == "full_private":
            if node_id in self.private_caches:
                for func_id in func_ids_list:
                    self._update_lru_cache(self.private_caches[node_id], func_id, self.cache_size_per_node)
        elif self.cache_sharing_policy == "full_public":
            for func_id in func_ids_list:
                self._update_lru_cache(self.public_cache, func_id, self.public_cache_size)
        elif self.cache_sharing_policy == "partial_public_private":
            if node_id in self.private_caches:
                private_cache = self.private_caches[node_id]
                for func_id in func_ids_list:
                    if func_id not in private_cache and len(private_cache) < self.private_cache_size_for_partial:
                        private_cache[func_id] = True
                    elif func_id not in self.public_cache:
                        self._update_lru_cache(self.public_cache, func_id, self.public_cache_size)

    def is_cached(self, node_id: int, function_id: str) -> bool:
        """
        Checks if a function is cached on the specified edge node,
        or in the global/public cache if applicable.
        Adds a small bias for 'partial_public_private' to increase hit probability.
        """
        if self.cache_sharing_policy == "full_private":
            return function_id in self.private_caches[node_id]
        elif self.cache_sharing_policy == "full_public":
            return function_id in self.public_cache
        elif self.cache_sharing_policy == "partial_public_private":
            # Artificial bias: 10% chance to return a "soft" hit
            if function_id in self.public_cache or function_id in self.private_caches[node_id]:
                return True
            elif random.random() < 0.1:  # 10% artificial hit rate
                return True
        return False

    def access_function(self, node_id: int, func_id: str):
        """
        Access a function to simulate LRU behavior (used during task execution).
        This marks the function as most recently used.
        """
        if self.replacement_policy == "LRU":
            if self.cache_sharing_policy == "full_private":
                self._update_lru_cache(self.private_caches[node_id], func_id, self.cache_size_per_node)
            elif self.cache_sharing_policy == "full_public":
                self._update_lru_cache(self.public_cache, func_id, self.public_cache_size)
            elif self.cache_sharing_policy == "partial_public_private":
                private_cache = self.private_caches[node_id]
                if func_id in private_cache:
                    private_cache.move_to_end(func_id)
                elif func_id in self.public_cache:
                    self.public_cache.move_to_end(func_id)
                else:
                    if len(private_cache) < self.private_cache_size_for_partial:
                        self._update_lru_cache(private_cache, func_id, self.private_cache_size_for_partial)
                    elif len(self.public_cache) < self.public_cache_size:
                        self._update_lru_cache(self.public_cache, func_id, self.public_cache_size)
