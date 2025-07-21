import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from fpdf import FPDF


def ensure_results_directory():
    """Ensure the results directory exists."""
    os.makedirs("results", exist_ok=True)


def plot_workflows_completed_on_time(results_list):
    """
    Bar chart showing how many workflows were completed on time for each policy.
    """
    ensure_results_directory()

    policy_names = [res["policy_name"] for res in results_list]
    on_time_counts = [res["workflows_completed_on_time"] for res in results_list]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=on_time_counts, palette="Blues_d", hue=policy_names, legend=False)
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
    ensure_results_directory()

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
    ensure_results_directory()

    for res in results_list:
        util_dict = res["edge_utilization_data"]
        policy_name = res["policy_name"]

        utilization_percentages = []
        for node_id in range(num_edge_nodes):
            if node_id in util_dict and len(util_dict[node_id]) > 0:
                utilization_percentages.append(100.0 * sum(util_dict[node_id]) / len(util_dict[node_id]))
            else:
                utilization_percentages.append(0.0)

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
    ensure_results_directory()

    df = pd.DataFrame(list(requests_dict.items()), columns=["NodeID", "RequestCount"])
    df = df.sort_values(by="RequestCount", ascending=False).reset_index(drop=True)

    # Handle edge case where all requests are 0
    if df["RequestCount"].max() == 0:
        print(f"Warning: No requests found for policy {policy_name}")
        return

    # Normalize colors
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    norm = plt.Normalize(df["RequestCount"].min(), df["RequestCount"].max())
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

    filename = f"results/edge_request_heatmap_{policy_name}.png"
    plt.savefig(filename)
    plt.close()


def plot_comparison_results(comparison_results):
    """
    Generates and saves comparison plots for all policy runs.
    """
    ensure_results_directory()

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

    # 3. Cold Starts per Policy (FIXED: was using on_time_counts instead of cold_starts)
    cold_starts = [res["total_cold_starts"] for res in comparison_results]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=cold_starts, hue=policy_names, legend=False, palette="Reds_d")
    plt.ylabel("Cold Starts")
    plt.xlabel("Policy")
    plt.title("Policy Comparison – Cold Start Counts")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/comparison_cold_starts.png")
    plt.close()

    # 4. Average Edge Utilization (FIXED: was using on_time_counts instead of avg_util)
    avg_util = []
    for res in comparison_results:
        util = res["edge_utilization_data"]
        flat_util = []
        for nid in util:
            if len(util[nid]) > 0:
                flat_util.append(sum(util[nid]) / len(util[nid]))
            else:
                flat_util.append(0.0)
        avg_util.append(round(np.mean(flat_util) * 100, 2))  # in %

    plt.figure(figsize=(10, 5))
    sns.barplot(x=policy_names, y=avg_util, hue=policy_names, legend=False, palette="Greens_d")
    plt.ylabel("Average Edge Utilization (%)")
    plt.xlabel("Policy")
    plt.title("Policy Comparison – Edge Utilization")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/comparison_edge_utilization.png")
    plt.close()

    # 5. Total Requests to Edge Nodes
    if len(comparison_results) > 0 and "edge_request_counts" in comparison_results[0]:
        total_requests = [sum(res["edge_request_counts"].values()) for res in comparison_results]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=policy_names, y=total_requests, palette="YlOrBr", hue=policy_names, legend=False)
        plt.ylabel("Total Requests to Edge Nodes")
        plt.xlabel("Policy")
        plt.title("Policy Comparison – Total Requests Sent to Edge")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("results/comparison_edge_requests.png")
        plt.close()


def generate_pdf_report_from_summary():
    """
    Generate a comprehensive PDF report from the simulation summary.
    """
    summary_path = "results/simulation_summary.csv"
    if not os.path.exists(summary_path):
        print("❌ 'simulation_summary.csv' not found. Skipping PDF report generation.")
        return

    try:
        df = pd.read_csv(summary_path)

        # Handle percentage column more robustly
        if "Avg Edge Utilization (%)" in df.columns:
            df["Avg Edge Utilization (%)"] = df["Avg Edge Utilization (%)"].astype(str).str.replace("%", "").astype(
                float)

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    class PDFReport(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, "FaaS Edge-Cloud Simulation Report", ln=True, align="C")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

        def section_title(self, title):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, title, ln=True)
            self.ln(2)

        def add_image(self, img_path, w=180):
            if os.path.exists(img_path):
                try:
                    self.image(img_path, w=w)
                    self.ln(10)
                except Exception as e:
                    self.set_font("Arial", "I", 10)
                    self.cell(0, 10, f"Error loading image: {img_path}", ln=True)
                    self.ln(5)
            else:
                self.set_font("Arial", "I", 10)
                self.cell(0, 10, f"Image not found: {img_path}", ln=True)
                self.ln(5)

    try:
        pdf = PDFReport()
        pdf.add_page()

        # Summary Table
        pdf.section_title("Simulation Summary Table")
        pdf.set_font("Arial", "", 8)  # Smaller font for table

        # Adjust column widths to fit better
        col_widths = [25, 20, 25, 22, 18, 18, 15, 22]
        header = ["Policy", "Workflows Sub", "Workflows Comp", "Workflows OnTime",
                  "Tasks Edge", "Tasks Cloud", "Cold Starts", "Avg Edge Util (%)"]

        for i, col in enumerate(header):
            pdf.cell(col_widths[i], 8, col, border=1, align="C")
        pdf.ln()

        for _, row in df.iterrows():
            row_data = [
                str(row["Policy"])[:20],  # Truncate long policy names
                str(row["Workflows Submitted"]),
                str(row["Workflows Completed"]),
                str(row["Workflows On Time"]),
                str(row["Tasks on Edge"]),
                str(row["Tasks on Cloud"]),
                str(row["Cold Starts"]),
                f'{row["Avg Edge Utilization (%)"]:.1f}%'
            ]
            for i, val in enumerate(row_data):
                pdf.cell(col_widths[i], 8, val, border=1, align="C")
            pdf.ln()

        # Add plots on new pages
        pdf.add_page()
        pdf.section_title("Workflows Completed On Time Comparison")
        pdf.add_image("results/comparison_workflows_on_time.png")

        pdf.section_title("Cold Starts Comparison")
        pdf.add_image("results/comparison_cold_starts.png")

        pdf.add_page()
        pdf.section_title("Edge Utilization Comparison")
        pdf.add_image("results/comparison_edge_utilization.png")

        pdf.section_title("Task Execution Distribution")
        pdf.add_image("results/comparison_task_execution_ratio.png")

        # Add edge requests comparison if it exists
        if os.path.exists("results/comparison_edge_requests.png"):
            pdf.add_page()
            pdf.section_title("Edge Requests Comparison")
            pdf.add_image("results/comparison_edge_requests.png")

        pdf.output("results/faas_simulation_report.pdf")
        print("📄 PDF report saved to: results/faas_simulation_report.pdf")

    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")


if __name__ == "__main__":
    # Test the PDF generation function
    generate_pdf_report_from_summary()