# src/config.py

import numpy as np
import networkx as nx
import random

# Simulation Parameters
NUM_EDGE_NODES = 12  # Your requested 10-15 nodes, let's use 12
EDGE_CACHE_SIZE_PER_NODE = 5
COLD_START_PENALTY = 0.5
SIMULATION_DURATION = 10000.0

# Network Configuration (for EdgeNetwork)
# --- Sparsely Connected Edge Graph ---
# Number of edges for the edge network (target 10-15)
NUM_EDGE_GRAPH_EDGES = 15 # You can change this between 10 and 15

# Generate a random graph with NUM_EDGE_NODES and NUM_EDGE_GRAPH_EDGES
# For reproducibility, set a seed *before* creating the graph
random.seed(42) # Using the same seed from main config
G_edge = nx.gnm_random_graph(NUM_EDGE_NODES, NUM_EDGE_GRAPH_EDGES)

# Convert the networkx graph to an adjacency matrix
ADJACENCY_MATRIX = nx.to_numpy_array(G_edge).tolist()

print(f"Config: Generated edge network with {NUM_EDGE_NODES} nodes and {G_edge.number_of_edges()} edges.")

# --- Latency Definitions ---
# Distance between adjacent edge nodes
EDGE_TO_EDGE_LATENCY = 5.0 # Fixed 2-10 units, let's pick 5
# Distance between Cloud and Edge nodes
CLOUD_TO_EDGE_LATENCY = 100.0 # Fixed 100 units

# Now, define which edge nodes are connected to the cloud.
# You requested "1-2 edges connected to cloud". Let's pick 2 for now.
# These IDs must be within [0, NUM_EDGE_NODES - 1].
EDGE_NODES_CONNECTED_TO_CLOUD = [0, 1] # Edge Node 0 and Edge Node 1 are connected to the Cloud.

# The Cloud's base latency from the perspective of a task being sent directly to it.
CLOUD_LATENCY = CLOUD_TO_EDGE_LATENCY

# The base_latency for EdgeNetwork.__init__
EDGE_LATENCY = EDGE_TO_EDGE_LATENCY

# Workflow Data
# Paths to your Pegasus XML datasets.
PEGASUS_WORKFLOW_FILEPATHS = [
    'data/Inspiral_30.xml',
    'data/CyberShake_30.xml',
    'data/Sipht_30.xml',
]

# Scheduling Policy
SCHEDULING_POLICY = "CriticalPathFirst"

# Cache Sharing Policy (This will be overridden by the test loop in simulator.py)
# Options: "full_private", "full_public", "partial_public_private"
CACHE_SHARING_POLICY = "full_private" # Default value

# If using "partial_public_private", define public cache size (as a fraction of total)
PUBLIC_CACHE_FRACTION = 0.2 # e.g., 20% of total cache capacity is public
# Predictive Caching Parameters
PREDICTION_INTERVAL = 50.0 # Time units between predictive cache loading runs (e.g., run every 50 time units)

# Workflow Submission Parameters (Phase 3 Goal 4)
# TOTAL_WORKFLOWS_TO_SIMULATE = 3 # Old fixed number
MIN_WORKFLOW_SUBMISSION_INTERVAL = 10.0 # Minimum time between workflow submissions
MAX_WORKFLOW_SUBMISSION_INTERVAL = 60.0 # Maximum time between workflow submissions
TOTAL_WORKFLOW_SUBMISSIONS = 20 # New: Total number of workflow instances to submit across the simulation
                                 # This will use workflows from PEGASUS_WORKFLOW_FILEPATHS in a round-robin or random fashion.

# Workflow Mix (Optional, for advanced scaling)
# You can define a probability distribution for selecting which workflow file to parse
# E.g., {'data/Inspiral_30.xml': 0.5, 'data/CyberShake_30.xml': 0.3, 'data/Sipht_30.xml': 0.2}
# If None, workflows are chosen uniformly or sequentially.
WORKFLOW_SELECTION_PROBABILITY = None


# Random seed for reproducibility
RANDOM_SEED = 42