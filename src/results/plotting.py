# src/results/plotting.py

import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Dict


def plot_comparison_results(comparison_results: List[Dict]):
    """
    Generates comparison plots for various simulation metrics across different policies.

    Args:
        comparison_results (list): A list of dictionaries, each containing results
                                   for a specific policy combination.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    policy_names = [res["policy_name"] for res in comparison_results]

    # --- Plot 1: Total Workflows Completed ---
    total_workflows = [res["total_workflows_completed"] for res in comparison_results]
    plt.figure(figsize=(10, 6))
    plt.bar(policy_names, total_workflows, color='skyblue')
    plt.xlabel("Policy Combination")
    plt.ylabel("Total Workflows Completed")
    plt.title("Comparison of Total Workflows Completed")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "total_workflows_completed.png"))
    plt.close()

    # --- Plot 2: Workflows Completed On Time ---
    workflows_on_time = [res["workflows_completed_on_time"] for res in comparison_results]
    plt.figure(figsize=(10, 6))
    plt.bar(policy_names, workflows_on_time, color='lightcoral')
    plt.xlabel("Policy Combination")
    plt.ylabel("Workflows Completed On Time")
    plt.title("Comparison of Workflows Completed On Time")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "workflows_completed_on_time.png"))
    plt.close()

    # --- Plot 3: Tasks on Edge vs. Cloud ---
    tasks_on_edge = [res["tasks_on_edge"] for res in comparison_results]
    tasks_on_cloud = [res["tasks_on_cloud"] for res in comparison_results]

    x = np.arange(len(policy_names))
    width = 0.35
    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, tasks_on_edge, width, label='Tasks on Edge', color='lightgreen')
    plt.bar(x + width/2, tasks_on_cloud, width, label='Tasks on Cloud', color='salmon')
    plt.xlabel("Policy Combination")
    plt.ylabel("Number of Tasks")
    plt.title("Comparison of Task Placement (Edge vs. Cloud)")
    plt.xticks(x, policy_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tasks_on_edge_vs_cloud.png"))
    plt.close()

    # --- Plot 4: Total Cold Starts ---
    total_cold_starts = [res["total_cold_starts"] for res in comparison_results]
    plt.figure(figsize=(10, 6))
    plt.bar(policy_names, total_cold_starts, color='gold')
    plt.xlabel("Policy Combination")
    plt.ylabel("Total Cold Starts")
    plt.title("Comparison of Total Cold Starts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "total_cold_starts.png"))
    plt.close()

    # --- Plot 5: Average Edge Utilization ---
    avg_edge_utilization_per_policy = []
    for res in comparison_results:
        total_util_sum = 0.0
        total_util_records = 0
        for node_id, util_list in res["edge_utilization_data"].items():
            total_util_sum += sum(util_list)
            total_util_records += len(util_list)
        if total_util_records > 0:
            avg_edge_utilization_per_policy.append(total_util_sum / total_util_records)
        else:
            avg_edge_utilization_per_policy.append(0.0) # Handle case with no utilization data

    plt.figure(figsize=(10, 6))
    plt.bar(policy_names, avg_edge_utilization_per_policy, color='mediumpurple')
    plt.xlabel("Policy Combination")
    plt.ylabel("Average Edge Utilization")
    plt.title("Comparison of Average Edge Utilization")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "average_edge_utilization.png"))
    plt.close()

    print(f"Comparison plots saved to the '{results_dir}' directory.")