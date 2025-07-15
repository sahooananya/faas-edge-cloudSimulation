from src.simulator import Simulator
from src.config import (
    NUM_EDGE_NODES, EDGE_CACHE_SIZE_PER_NODE, COLD_START_PENALTY,
    EDGE_TO_EDGE_LATENCY, CLOUD_TO_EDGE_LATENCY, PEGASUS_WORKFLOW_FILEPATHS,
    ADJACENCY_MATRIX, PUBLIC_CACHE_FRACTION,
)
from src.results import plotting

policy_combinations = [
    ("EDF", "full_public"),
    ("EDF", "partial_public_private"),
    ("EDF", "full_private"),
    ("CriticalPathFirst", "full_public"),
    ("CriticalPathFirst", "partial_public_private"),
    ("CriticalPathFirst", "full_private"),
]

comparison_results = []

for sched_policy, cache_policy in policy_combinations:
    print(f"\n=== Running: Scheduling={sched_policy}, Cache={cache_policy} ===")

    simulator = Simulator(
        num_edge_nodes=NUM_EDGE_NODES,
        cache_size=EDGE_CACHE_SIZE_PER_NODE,
        cold_start_penalty=COLD_START_PENALTY,
        edge_latency=EDGE_TO_EDGE_LATENCY,
        cloud_latency=CLOUD_TO_EDGE_LATENCY,
        workflow_filepaths=PEGASUS_WORKFLOW_FILEPATHS,
        adjacency_matrix=ADJACENCY_MATRIX,
        scheduling_policy=sched_policy,
        cache_sharing_policy=cache_policy,
        public_cache_fraction=PUBLIC_CACHE_FRACTION
    )

    simulator.run_simulation()


    comparison_results.append({
        "policy_name": f"{sched_policy}_{cache_policy}",
        "total_workflows_completed": simulator.global_total_workflows_completed,
        "workflows_completed_on_time": simulator.global_workflows_completed_on_time,
        "tasks_on_edge": simulator.global_tasks_on_edge,
        "tasks_on_cloud": simulator.global_tasks_on_cloud,
        "total_cold_starts": simulator.global_cold_starts,
        "total_simulation_time": simulator.current_time,
        "edge_utilization_data": simulator.global_edge_utilization,
        "workflow_stats_list": list(simulator.workflow_stats.values())
    })


print("\n=== Generating Final Comparison Plots ===")
plotting.plot_comparison_results(comparison_results)
print("Comparison plots generated in the 'results' directory.")
