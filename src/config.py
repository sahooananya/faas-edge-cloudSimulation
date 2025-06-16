'''
import random


import networkx as nx

# Conversion factor: 1 second = 1000 milliseconds
MS_PER_SECOND = 1000.0
DEFAULT_FUNCTION_RUNTIME = 0.1 * MS_PER_SECOND
# Simulation Parameters
NUM_EDGE_NODES = 15
EDGE_CACHE_SIZE_PER_NODE = 6  # Increased for fewer cold starts
COLD_START_PENALTY = 0.05 * MS_PER_SECOND  # 0.5 seconds = 500 ms penalty
SIMULATION_DURATION = 3 000 000 * MS_PER_SECOND  # 300,000 seconds in ms

# Network Configuration
NUM_EDGE_GRAPH_EDGES = 20
random.seed(42)
G_edge = nx.gnm_random_graph(NUM_EDGE_NODES, NUM_EDGE_GRAPH_EDGES)
ADJACENCY_MATRIX = nx.to_numpy_array(G_edge).tolist()

print(f"Config: Generated edge network with {NUM_EDGE_NODES} nodes and {G_edge.number_of_edges()} edges.")

# Latency
EDGE_TO_EDGE_LATENCY = 5.0 * MS_PER_SECOND  # 5 seconds
CLOUD_TO_EDGE_LATENCY = 100.0 * MS_PER_SECOND  # 100 seconds
EDGE_NODES_CONNECTED_TO_CLOUD = [0, 1]

CLOUD_LATENCY = CLOUD_TO_EDGE_LATENCY
EDGE_LATENCY = EDGE_TO_EDGE_LATENCY

# Workflows
PEGASUS_WORKFLOW_FILEPATHS = [
    #'data/Inspiral_30.xml',
    'data/CyberShake_30.xml',
    #'data/Sipht_30.xml',
]

# Policies
SCHEDULING_POLICY = "CriticalPathFirst"
CACHE_SHARING_POLICY = "partial_public_private"  # Optimized for max reuse
PUBLIC_CACHE_FRACTION = 0.8

# Prediction Interval
PREDICTION_INTERVAL = 20.0 * MS_PER_SECOND

# Workflow Submission
MIN_WORKFLOW_SUBMISSION_INTERVAL = 10.0 * MS_PER_SECOND
MAX_WORKFLOW_SUBMISSION_INTERVAL = 60.0 * MS_PER_SECOND
TOTAL_WORKFLOW_SUBMISSIONS = 50

WORKFLOW_SELECTION_PROBABILITY = None
RANDOM_SEED = 42
'''

import random
import networkx as nx

# Units
MS_PER_SECOND = 1000.0
DEFAULT_FUNCTION_RUNTIME = 0.1 * MS_PER_SECOND
# --- Simulation Parameters ---
NUM_EDGE_NODES = 15
EDGE_CACHE_SIZE_PER_NODE = 8         # Larger edge cache for higher hit rate
COLD_START_PENALTY = 0.1 * MS_PER_SECOND  # 10 ms cold start → favors edge
SIMULATION_DURATION = 300000 * MS_PER_SECOND  # 5 minutes

# --- Edge Network ---
NUM_EDGE_GRAPH_EDGES = 30            # More connectivity boosts scheduling flexibility
random.seed(42)
G_edge = nx.gnm_random_graph(NUM_EDGE_NODES, NUM_EDGE_GRAPH_EDGES)
ADJACENCY_MATRIX = nx.to_numpy_array(G_edge).tolist()
print(f"Config: Generated edge network with {NUM_EDGE_NODES} nodes and {G_edge.number_of_edges()} edges.")

# --- Latency ---
EDGE_TO_EDGE_LATENCY = 0.01 * MS_PER_SECOND   # 10 ms edge latency
CLOUD_TO_EDGE_LATENCY = 0.15 * MS_PER_SECOND  # 150 ms cloud delay → discourage cloud
EDGE_NODES_CONNECTED_TO_CLOUD = [0, 1]

# --- Workflows ---
PEGASUS_WORKFLOW_FILEPATHS = [
    'data/CyberShake_30.xml',
    'data/Inspiral_30.xml',
    'data/Sipht_30.xml',
]

# --- Policies ---
SCHEDULING_POLICY = "CriticalPathFirst"       # Good for prioritizing tight-deadline tasks
CACHE_SHARING_POLICY = "partial_public_private"
PUBLIC_CACHE_FRACTION = 0.6

# --- Scheduling Behavior ---
PREDICTION_INTERVAL = 5.0 * MS_PER_SECOND
MIN_WORKFLOW_SUBMISSION_INTERVAL = 15.0 * MS_PER_SECOND
MAX_WORKFLOW_SUBMISSION_INTERVAL = 30.0 * MS_PER_SECOND
TOTAL_WORKFLOW_SUBMISSIONS = 50
WORKFLOW_SELECTION_PROBABILITY = None

# --- Misc ---
RANDOM_SEED = 42

