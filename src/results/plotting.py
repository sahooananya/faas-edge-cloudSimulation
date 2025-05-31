import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os  # For ensuring output directory exists


def plot_workflow_completion(workflow_stats_list):
    """Visualizes the count of workflows completed on time vs. delayed."""
    if not workflow_stats_list:
        print("No workflow stats to plot for completion status.")
        return

    on_time_count = sum(1 for stats in workflow_stats_list if stats.workflow_completed_within_deadline)
    delayed_count = len(workflow_stats_list) - on_time_count

    labels = ['Completed On Time', 'Delayed']
    sizes = [on_time_count, delayed_count]
    colors = ['#66b31a', '#ff9999']  # Green for on time, red for delayed
    explode = (0.1, 0)  # explode the 'on time' slice for emphasis

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Workflow Completion Status (On Time vs. Delayed)')

    # Ensure logs directory exists
    output_dir = 'logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'workflow_completion_status.png'))
    plt.close()  # Close plot to free memory


def plot_edge_cloud_task_ratio(workflow_stats_list):
    """Visualizes the ratio of tasks executed on Edge vs. Cloud."""
    if not workflow_stats_list:
        print("No workflow stats to plot for Edge vs. Cloud ratio.")
        return

    total_edge_tasks = sum(stats.tasks_executed_on_edge for stats in workflow_stats_list)
    total_cloud_tasks = sum(stats.tasks_executed_on_cloud for stats in workflow_stats_list)

    if total_edge_tasks == 0 and total_cloud_tasks == 0:
        print("No tasks executed to plot Edge vs. Cloud ratio.")
        return

    labels = ['Edge Tasks', 'Cloud Tasks']
    sizes = [total_edge_tasks, total_cloud_tasks]
    colors = ['#ffcc99', '#66b3ff']  # Orange for edge, blue for cloud
    explode = (0.1, 0)

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Task Execution Location: Edge vs. Cloud')

    output_dir = 'logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'edge_cloud_task_ratio.png'))
    plt.close()


def plot_edge_utilization_heatmap(edge_utilization_data, num_edge_nodes, simulation_duration):
    """Visualizes the utilization of each edge node as a heatmap or bar chart."""
    if not edge_utilization_data or simulation_duration == 0:
        print("No edge utilization data or simulation duration to plot.")
        return

    utilization_percentages = []
    node_labels = []
    for i in range(num_edge_nodes):
        node_id = i  # Assuming node IDs are 0 to num_edge_nodes-1
        util_time = edge_utilization_data.get(node_id, 0.0)
        percentage = (util_time / simulation_duration) * 100
        utilization_percentages.append(percentage)
        node_labels.append(f'Edge {node_id}')

    # For a few nodes, a bar plot is more intuitive than a 1xN heatmap
    plt.figure(figsize=(max(6, num_edge_nodes * 0.8), 5))  # Adjust figure size dynamically
    # Fix for FutureWarning: Passing `palette` without assigning `hue` is deprecated
    sns.barplot(x=node_labels, y=utilization_percentages, hue=node_labels, palette='viridis', legend=False)
    plt.ylabel('Utilization (%)')
    plt.xlabel('Edge Node ID')
    plt.title('Edge Node Utilization')
    plt.ylim(0, 100)  # Utilization is always between 0 and 100%
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels overlapping

    output_dir = 'logs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'edge_utilization.png'))
    plt.close()


def plot_policy_comparison(comparison_results: list, num_edge_nodes: int):
    """
    Plots a comparison of key metrics across different cache sharing policies.
    """
    if not comparison_results:
        print("No comparison results to plot.")
        return

    policies = [res['policy_name'] for res in comparison_results]

    # Data for plots
    workflow_completion_rates = []
    edge_task_ratios = []
    total_cold_starts = []
    avg_edge_utilizations = []  # Calculate this here

    for res in comparison_results:
        total_wf = res['total_workflows_completed']
        on_time_wf = res['workflows_completed_on_time']
        workflow_completion_rates.append((on_time_wf / total_wf) * 100 if total_wf > 0 else 0)

        total_tasks = res['tasks_on_edge'] + res['tasks_on_cloud']
        edge_task_ratios.append((res['tasks_on_edge'] / total_tasks) * 100 if total_tasks > 0 else 0)

        total_cold_starts.append(res['total_cold_starts'])

        # Calculate average edge utilization for this policy
        total_util_time = sum(res['edge_utilization_data'].values())
        sim_duration = res['total_simulation_time']
        avg_util_percent = (total_util_time / (
                    sim_duration * num_edge_nodes)) * 100 if sim_duration > 0 and num_edge_nodes > 0 else 0
        avg_edge_utilizations.append(avg_util_percent)

    # --- Plot 1: Workflow Completion Rate ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=policies, y=workflow_completion_rates, palette='viridis')
    plt.ylabel('Workflows Completed On Time (%)')
    plt.title('Workflow Completion Rate by Cache Sharing Policy')
    plt.ylim(0, 100)
    output_dir = 'logs'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'policy_comparison_workflow_completion.png'))
    plt.close()

    # --- Plot 2: Edge Task Ratio ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=policies, y=edge_task_ratios, palette='magma')
    plt.ylabel('Tasks Executed on Edge (%)')
    plt.title('Edge Task Execution Ratio by Cache Sharing Policy')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'policy_comparison_edge_ratio.png'))
    plt.close()

    # --- Plot 3: Total Cold Starts ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=policies, y=total_cold_starts, palette='plasma')
    plt.ylabel('Total Cold Starts')
    plt.title('Total Cold Starts by Cache Sharing Policy')
    plt.savefig(os.path.join(output_dir, 'policy_comparison_cold_starts.png'))
    plt.close()

    # --- Plot 4: Average Edge Utilization ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=policies, y=avg_edge_utilizations, palette='cubehelix')
    plt.ylabel('Average Edge Node Utilization (%)')
    plt.title('Average Edge Node Utilization by Cache Sharing Policy')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'policy_comparison_avg_edge_utilization.png'))
    plt.close()

    print("Comparison plots generated successfully!")