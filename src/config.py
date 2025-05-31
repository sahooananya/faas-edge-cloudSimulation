# src/config.py

import numpy as np
import networkx as nx
import random

# Conversion factor: 1 second = 1000 milliseconds
MS_PER_SECOND = 1000.0

# Simulation Parameters
NUM_EDGE_NODES = 15  # Max 15 nodes as per your instruction
EDGE_CACHE_SIZE_PER_NODE = 15  # Keep large for good cold start performance
COLD_START_PENALTY = 0.05 * MS_PER_SECOND  # 0.5 seconds = 500 ms penalty
SIMULATION_DURATION = 300000.0 * MS_PER_SECOND  # 300,000 seconds = 300,000,000 ms (Massively increased duration to allow for huge latencies)

# Network Configuration (for EdgeNetwork)
NUM_EDGE_GRAPH_EDGES = 20  # Adjusted for 15 nodes, to maintain connectivity (was 25, changed to 20)

random.seed(42)
G_edge = nx.gnm_random_graph(NUM_EDGE_NODES, NUM_EDGE_GRAPH_EDGES)
ADJACENCY_MATRIX = nx.to_numpy_array(G_edge).tolist()

print(f"Config: Generated edge network with {NUM_EDGE_NODES} nodes and {G_edge.number_of_edges()} edges.")

# Latency Definitions (in milliseconds) - AS PER YOUR EXPLICIT INSTRUCTIONS
EDGE_TO_EDGE_LATENCY = 5.0 * MS_PER_SECOND  # 5 seconds = 5000 ms (within 2-10 sec range)
CLOUD_TO_EDGE_LATENCY = 100.0 * MS_PER_SECOND  # 100 seconds = 100,000 ms

EDGE_NODES_CONNECTED_TO_CLOUD = [0, 1]

CLOUD_LATENCY = CLOUD_TO_EDGE_LATENCY
EDGE_LATENCY = EDGE_TO_EDGE_LATENCY

# Workflow Data
PEGASUS_WORKFLOW_FILEPATHS = [
    'data/Inspiral_30.xml',
    'data/CyberShake_30.xml',
    'data/Sipht_30.xml',
]

# Scheduling Policy
SCHEDULING_POLICY = "CriticalPathFirst"

# Cache Sharing Policy
CACHE_SHARING_POLICY = "partial_public_private"  # Default, will be overridden by loop
PUBLIC_CACHE_FRACTION = 0.2

# Predictive Caching Parameters
PREDICTION_INTERVAL = 20.0 * MS_PER_SECOND  # 20 seconds = 20,000 ms between predictions

# Workflow Submission Parameters
MIN_WORKFLOW_SUBMISSION_INTERVAL = 10.0 * MS_PER_SECOND  # 10 seconds = 10,000 ms
MAX_WORKFLOW_SUBMISSION_INTERVAL = 60.0 * MS_PER_SECOND  # 60 seconds = 60,000 ms
TOTAL_WORKFLOW_SUBMISSIONS = 50  # Keep at 50 for now

WORKFLOW_SELECTION_PROBABILITY = None

# Random seed for reproducibility
RANDOM_SEED = 42