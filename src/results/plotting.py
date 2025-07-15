import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def plot_workflows_completed_on_time(results_list):
    """
    Bar chart showing how many workflows were completed on time for each policy.
    """
    policy_names = [res["policy_name"] for res in results_list]
    on_time_counts = [res["workflows_completed_on_time"] for res in results_list]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=on_time_counts, palette="Blues_d",hue=policy_names, legend=False)
    plt.ylabel("Workflows Completed On Time")
    plt.xlabel("Policy")
    plt.title("Workflows Completed On Time per Policy")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/workflows_completed_on_time.png")
    plt.close()


def plot_task_execution_ratio(results_list):
    """
    Stacked bar chart: ratio of tasks executed on edge vs cloud.
    """
    policy_names = [res["policy_name"] for res in results_list]
    edge_tasks = [res["tasks_on_edge"] for res in results_list]
    cloud_tasks = [res["tasks_on_cloud"] for res in results_list]

    x = np.arange(len(policy_names))

    plt.figure(figsize=(10, 5))
    plt.bar(x, edge_tasks, label="Edge", color='skyblue')
    plt.bar(x, cloud_tasks, bottom=edge_tasks, label="Cloud", color='lightgray')
    plt.xticks(x, policy_names, rotation=30)
    plt.ylabel("Number of Tasks Executed")
    plt.title("Edge vs Cloud Task Execution per Policy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/task_execution_ratio.png")
    plt.close()


def plot_edge_utilization_heatmap(results_list, num_edge_nodes, simulation_duration):
    """
    Heatmap showing edge utilization % per node.
    """
    for res in results_list:
        util_dict = res["edge_utilization_data"]
        policy_name = res["policy_name"]

        utilization_percentages = [
            100.0 * sum(util_dict[node_id]) / len(util_dict[node_id])
            if len(util_dict[node_id]) > 0 else 0.0
            for node_id in range(num_edge_nodes)
        ]

        size = int(np.ceil(np.sqrt(num_edge_nodes)))
        padded = np.zeros((size * size,))
        padded[:len(utilization_percentages)] = utilization_percentages
        matrix = padded.reshape((size, size))

        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm", cbar=True)
        plt.title(f"Edge Node Utilization Heatmap: {policy_name}")
        plt.xlabel("Node Grid X")
        plt.ylabel("Node Grid Y")
        plt.tight_layout()
        plt.savefig(f"results/utilization_heatmap_{policy_name}.png")
        plt.close()


def plot_edge_request_heatmap(requests_dict: dict, num_nodes: int, policy_name: str = "default_policy"):
    """
    Saves a vertical bar heatmap showing the number of task requests to each edge node.
    """
    df = pd.DataFrame(list(requests_dict.items()), columns=["NodeID", "RequestCount"])
    df = df.sort_values(by="RequestCount", ascending=False).reset_index(drop=True)

    # Normalize colors
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    norm = plt.Normalize(df["RequestCount"].min(), df["RequestCount"].max() or 1)
    colors = [cmap(norm(val)) for val in df["RequestCount"]]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df["NodeID"], df["RequestCount"], color=colors)

    for bar, val in zip(bars, df["RequestCount"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val), ha='center', va='bottom', fontsize=9)

    plt.title(f"Edge Node Task Request Distribution\n({policy_name})", fontsize=14, fontweight='bold')
    plt.xlabel("Edge Node ID")
    plt.ylabel("Number of Requests")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    filename = f"results/edge_request_heatmap_{policy_name}.png"
    plt.savefig(filename)
    plt.close()


def plot_comparison_results(comparison_results):
    """
    Generates and saves comparison plots for all policy runs.
    """
    os.makedirs("results", exist_ok=True)

    policy_names = [res["policy_name"] for res in comparison_results]

    # 1. Workflows Completed On Time
    on_time_counts = [res["workflows_completed_on_time"] for res in comparison_results]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=on_time_counts, hue=policy_names, legend=False, palette="viridis")

    plt.ylabel("Workflows Completed On Time")
    plt.xlabel("Policy")
    plt.title("Policy Comparison – Workflows Completed On Time")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/comparison_workflows_on_time.png")
    plt.close()

    # 2. Edge vs Cloud Task Execution
    edge_tasks = [res["tasks_on_edge"] for res in comparison_results]
    cloud_tasks = [res["tasks_on_cloud"] for res in comparison_results]
    x = np.arange(len(policy_names))
    plt.figure(figsize=(10, 5))
    plt.bar(x, edge_tasks, label="Edge", color='skyblue')
    plt.bar(x, cloud_tasks, bottom=edge_tasks, label="Cloud", color='lightgray')
    plt.xticks(x, policy_names, rotation=30)
    plt.ylabel("Number of Tasks")
    plt.title("Policy Comparison – Edge vs Cloud Task Execution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/comparison_task_execution_ratio.png")
    plt.close()

    # 3. Cold Starts per Policy
    cold_starts = [res["total_cold_starts"] for res in comparison_results]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=on_time_counts, hue=policy_names, legend=False, palette="viridis")
    plt.ylabel("Cold Starts")
    plt.xlabel("Policy")
    plt.title("Policy Comparison – Cold Start Counts")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/comparison_cold_starts.png")
    plt.close()

    # 4. Average Edge Utilization
    avg_util = []
    for res in comparison_results:
        util = res["edge_utilization_data"]
        flat_util = [sum(util[nid]) / len(util[nid]) if len(util[nid]) > 0 else 0 for nid in util]
        avg_util.append(round(np.mean(flat_util) * 100, 2))  # in %

    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=on_time_counts, hue=policy_names, legend=False, palette="viridis")
    plt.ylabel("Average Edge Utilization (%)")
    plt.xlabel("Policy")
    plt.title("Policy Comparison – Edge Utilization")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/comparison_edge_utilization.png")
    plt.close()

    # 5. Total Requests to Edge Nodes
    if "edge_request_counts" in comparison_results[0]:  # only if present
        total_requests = [sum(res["edge_request_counts"].values()) for res in comparison_results]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=policy_names, y=total_requests, palette="YlOrBr",hue=policy_names, legend=False)
        plt.ylabel("Total Requests to Edge Nodes")
        plt.xlabel("Policy")
        plt.title("Policy Comparison – Total Requests Sent to Edge")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("results/comparison_edge_requests.png")
        plt.close()
