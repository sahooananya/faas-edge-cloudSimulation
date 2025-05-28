# src/config.py

import numpy as np

# Simulation Parameters
NUM_EDGE_NODES = 3  # Number of edge servers
EDGE_CACHE_SIZE_PER_NODE = 5  # Number of functions an edge node can cache
COLD_START_PENALTY = 2.0  # Time units penalty for a cold start at edge
EDGE_LATENCY = 0.01  # Base latency for edge processing (e.g., communication to local scheduler)
CLOUD_LATENCY = 0.1  # Base latency for cloud access (e.g., network round-trip to cloud)
SIMULATION_DURATION = 10000.0 # Maximum simulation time (helps prevent infinite loops)

# Workflow Data
PEGASUS_WORKFLOW_FILEPATHS = [
    'data/CyberShake_30.xml',
    'data/Inspiral_30.xml',
    'data/Sipht_30.xml',
    # Add more workflow files as needed for testing
]

# Scheduling Policy
SCHEDULING_POLICY = "EDF"

# Cache Sharing Policy (for Phase 1, only full_private is relevant for LRU at edge)
# Options will be expanded in Phase 2
CACHE_SHARING_POLICY = "full_private"

# Adjacency Matrix for Edge Network (for Phase 2)
# For Phase 1, this will be a placeholder, connectivity is not used yet.
ADJACENCY_MATRIX = np.ones((NUM_EDGE_NODES, NUM_EDGE_NODES)).tolist()
for i in range(NUM_EDGE_NODES):
    ADJACENCY_MATRIX[i][i] = 0

# Random seed for reproducibility (optional)
RANDOM_SEED = 42