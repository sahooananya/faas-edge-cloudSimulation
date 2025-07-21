# import csv
# import heapq
# import os
# import random
# import time
# from typing import Dict
# from typing import List
#
# from .cache_manager import CacheManager
# from .config import (
#     SIMULATION_DURATION, RANDOM_SEED, PREDICTION_INTERVAL,
#     MIN_WORKFLOW_SUBMISSION_INTERVAL, MAX_WORKFLOW_SUBMISSION_INTERVAL,
#     TOTAL_WORKFLOW_SUBMISSIONS
# )
# from .edge_network import EdgeNetwork
# from .models import Task, Workflow, Cloud, WorkflowStats
# from .parser import PegasusWorkflowParser
# from .scheduler import Scheduler
#
#
# class Simulator:
#     def __init__(self, num_edge_nodes, cache_size, cold_start_penalty,
#                  edge_latency, cloud_latency, workflow_filepaths,
#                  adjacency_matrix, scheduling_policy, cache_sharing_policy,
#                  public_cache_fraction=0.0):
#
#         random.seed(RANDOM_SEED)
#         self.current_time = 0.0
#         self.event_queue = []
#         self._event_id_counter = 0
#
#         self.num_edge_nodes = num_edge_nodes
#         self.cache_size = cache_size
#         self.cold_start_penalty = cold_start_penalty
#         self.edge_latency = edge_latency
#         self.cloud_latency = cloud_latency
#         self.workflow_filepaths = workflow_filepaths
#         self.adjacency_matrix = adjacency_matrix
#         self.scheduling_policy = scheduling_policy
#         self.cache_sharing_policy = cache_sharing_policy
#         self.public_cache_fraction = public_cache_fraction
#
#         self.edge_network = EdgeNetwork(num_edge_nodes, edge_latency, adjacency_matrix)
#         for node in self.edge_network.edge_nodes:
#             node.capacity = 1
#             node.cache_size = self.cache_size
#
#         self.cloud = Cloud(base_latency=self.cloud_latency)
#         self.cache_manager = CacheManager(cache_size, num_edge_nodes, "LRU", cache_sharing_policy,
#                                           public_cache_fraction)
#         self.scheduler = Scheduler(self.edge_network, self.cloud, self.cache_manager, cold_start_penalty,
#                                    scheduling_policy)
#         self.parser = PegasusWorkflowParser()
#
#         self.workflow_templates: Dict[str, Workflow] = {}
#         self.workflow_stats: Dict[str, WorkflowStats] = {}
#
#
#         self.global_total_workflows_completed = 0
#         self.global_workflows_completed_on_time = 0
#         self.global_tasks_on_edge = 0
#         self.global_tasks_on_cloud = 0
#         self.global_cold_starts = 0
#         self.global_edge_utilization: Dict[int, List[int]] = {i: [] for i in range(self.num_edge_nodes)}
#         self.last_utilization_record_time = 0.0
#
#         self._load_workflow_templates()
#         self._schedule_initial_events()
#
#         print(
#             f"Simulator initialized with scheduling policy: {self.scheduling_policy} and cache policy: {self.cache_sharing_policy}")
#
#     def _add_event(self, time: float, event_type: str, *args):
#         """Adds an event to the event queue."""
#         heapq.heappush(self.event_queue, (time, self._event_id_counter, event_type, args))
#         self._event_id_counter += 1
#
#     def _load_workflow_templates(self):
#         """Loads all workflow templates using the parser."""
#         for filepath in self.workflow_filepaths:
#             workflow_template = self.parser.parse_workflow_template(filepath)
#             if workflow_template:
#                 self.workflow_templates[workflow_template.name] = workflow_template
#             else:
#                 print(f"Warning: Failed to load workflow template from {filepath}")
#
#     def _schedule_initial_events(self):
#         """Schedules initial workflow submissions and other periodic events."""
#         available_workflow_names = list(self.workflow_templates.keys())
#         if not available_workflow_names:
#             print("No workflow templates loaded. Cannot schedule submission events.")
#             return
#
#         current_event_time = 0.0
#         for i in range(TOTAL_WORKFLOW_SUBMISSIONS):
#             workflow_name_to_submit = random.choice(available_workflow_names)
#             submission_interval = random.uniform(MIN_WORKFLOW_SUBMISSION_INTERVAL, MAX_WORKFLOW_SUBMISSION_INTERVAL)
#             current_event_time += submission_interval
#             self._add_event(current_event_time, "workflow_submission", workflow_name_to_submit)
#
#         print(f"Generated {TOTAL_WORKFLOW_SUBMISSIONS} workflow submission events.")
#
#
#         self._add_event(PREDICTION_INTERVAL, "predictive_prefetch")
#
#
#         self._add_event(SIMULATION_DURATION + 1, "simulation_end")
#     def _process_workflow_submission(self, workflow_name: str):
#         """Processes a workflow submission event."""
#         workflow_template = self.workflow_templates.get(workflow_name)
#         if not workflow_template:
#             print(f"Error: Workflow template '{workflow_name}' not found for submission.")
#             return
#
#         workflow_instance_id = f"{workflow_template.id}_inst_{int(self.current_time)}_{random.randint(0, 9999)}"
#
#         workflow_instance = workflow_template.create_instance(
#             new_id=workflow_instance_id,
#             current_simulation_time=self.current_time
#         )
#
#         self.workflow_stats[workflow_instance.id] = WorkflowStats(workflow_instance.id, workflow_instance.name,
#                                                                   workflow_instance.total_tasks)
#
#         self.workflow_stats[workflow_instance.id].workflow_instance = workflow_instance
#
#         for task in workflow_instance.get_ready_tasks():
#             task.ready_time = self.current_time
#             self.scheduler.add_ready_task(task)
#
#     def _process_task_completion(self, task: Task):
#         """Handles a task completion event."""
#         workflow_instance = task.workflow_instance
#         if not workflow_instance:
#             print(f"Error: Workflow instance not found for task {task.id}.")
#             return
#
#         self.global_cold_starts += 1 if task.cold_start_penalty > 0 else 0
#         if task.assigned_node_type == "edge":
#             self.global_tasks_on_edge += 1
#         elif task.assigned_node_type == "cloud":
#             self.global_tasks_on_cloud += 1
#
#         self.workflow_stats[workflow_instance.id].add_task_execution_detail(
#             task.assigned_node_type,
#             task.end_time <= task.deadline,
#             task.wait_time,
#             task.cold_start_penalty
#         )
#
#         workflow_instance.mark_task_completed(task.id)
#
#
#         for next_task in workflow_instance.get_successors(task.id):
#             all_deps_met = all(dep_id in workflow_instance.completed_tasks for dep_id in next_task.dependencies)
#
#             if all_deps_met and next_task.start_time is None:
#                 next_task.ready_time = self.current_time
#                 self.scheduler.add_ready_task(next_task)
#
#         if workflow_instance.is_completed():
#             self.global_total_workflows_completed += 1
#             workflow_instance.end_time = self.current_time
#
#             wf_stats_obj = self.workflow_stats.get(workflow_instance.id)
#             if wf_stats_obj:
#                 wf_stats_obj.workflow_completed_within_deadline = workflow_instance.check_if_completed_within_deadline(
#                     self.current_time)
#                 if wf_stats_obj.workflow_completed_within_deadline:
#                     self.global_workflows_completed_on_time += 1
#
#     def _process_predictive_prefetch(self):
#         """Triggers predictive caching in the scheduler."""
#         # This line is correctly calling predictive_prefetch with two arguments
#         self.scheduler.predictive_prefetch(self.workflow_stats, self.current_time)
#         # Schedule the next prediction event
#         self._add_event(self.current_time + PREDICTION_INTERVAL, "predictive_prefetch")
#
#     def _record_edge_utilization(self):
#         """Records the busy/idle status of each edge node at the current time."""
#         for node in self.edge_network.get_all_edge_nodes():
#             is_busy = 1 if node.is_busy(self.current_time) else 0
#             self.global_edge_utilization[node.id].append(is_busy)
#
#     def run_simulation(self):
#         """Runs the main simulation loop."""
#         start_time_real = time.time()
#         print(f"Starting simulation for {SIMULATION_DURATION / 1000 / 60:.2f} minutes...")
#
#         all_function_ids = list(self.parser.functions_catalog.keys())
#         self.scheduler.initial_prefetch(all_function_ids)
#
#         self.last_utilization_record_time = 0.0
#
#         while self.event_queue and self.current_time <= SIMULATION_DURATION:
#             event_time, _, event_type, event_args = heapq.heappop(self.event_queue)
#
#             if event_time > self.current_time:
#                 self.current_time = event_time
#                 while self.last_utilization_record_time + PREDICTION_INTERVAL <= self.current_time:
#                     self._record_edge_utilization()
#                     self.last_utilization_record_time += PREDICTION_INTERVAL
#
#             if self.current_time > SIMULATION_DURATION:
#                 break
#
#             if event_type == "workflow_submission":
#                 self._process_workflow_submission(*event_args)
#             elif event_type == "task_completion":
#                 self._process_task_completion(*event_args)
#             elif event_type == "predictive_prefetch":
#                 self._process_predictive_prefetch()
#             elif event_type == "simulation_end":
#                 print(
#                     f"Simulation ended at {self.current_time / 1000:.2f} seconds ({self.current_time / 1000 / 60:.2f} minutes).")
#                 break
#
#             scheduled_tasks = self.scheduler.schedule_tasks(self.current_time, self.workflow_stats)
#             for task, assigned_node, assigned_node_type, cold_start_penalty, wait_time in scheduled_tasks:
#                 task.start_time = self.current_time + wait_time
#                 task.end_time = task.start_time + task.runtime
#                 task.assigned_node = assigned_node
#                 task.assigned_node_type = assigned_node_type
#                 task.cold_start_penalty = cold_start_penalty
#                 task.wait_time = wait_time
#
#                 self._add_event(task.end_time, "task_completion", task)
#
#         while self.last_utilization_record_time + PREDICTION_INTERVAL <= self.current_time:
#             self._record_edge_utilization()
#             self.last_utilization_record_time += PREDICTION_INTERVAL
#
#         end_time_real = time.time()
#         print(f"Total simulation wall time: {end_time_real - start_time_real:.2f} seconds.")
#
#         self._save_results(
#             policy_name=f"{self.scheduling_policy}_{self.cache_sharing_policy}",
#             total_workflows_submitted=TOTAL_WORKFLOW_SUBMISSIONS,
#             total_workflows_completed=self.global_total_workflows_completed,
#             workflows_completed_on_time=self.global_workflows_completed_on_time,
#             tasks_on_edge=self.global_tasks_on_edge,
#             tasks_on_cloud=self.global_tasks_on_cloud,
#             cold_starts=self.global_cold_starts,
#             edge_utilization_data=self.global_edge_utilization,
#             workflow_stats=list(self.workflow_stats.values())
#         )
#
#     def _save_results(self, policy_name: str, total_workflows_submitted: int,
#                       total_workflows_completed: int, workflows_completed_on_time: int,
#                       tasks_on_edge: int, tasks_on_cloud: int, cold_starts: int,
#                       edge_utilization_data: Dict[int, List[int]],
#                       workflow_stats: List[WorkflowStats]):
#         """Save simulation results to CSV files and prints to console."""
#         results_dir = "results"
#         os.makedirs(results_dir, exist_ok=True)
#
#         summary_file = os.path.join(results_dir, "simulation_summary.csv")
#         per_workflow_file = os.path.join(results_dir, "per_workflow_stats.csv")
#
#         total_busy_intervals = 0
#         total_intervals_recorded = 0
#         for node_id, util_list in edge_utilization_data.items():
#             total_busy_intervals += sum(util_list)
#             total_intervals_recorded += len(util_list)
#
#         avg_util_percentage = (
#                                           total_busy_intervals / total_intervals_recorded) * 100 if total_intervals_recorded > 0 else 0.0
#
#
#         print(f"\n=== Simulation Results ({policy_name}) ===")
#         print(f"Total Workflows Submitted: {total_workflows_submitted}")
#         print(f"Total Workflows Completed: {total_workflows_completed}")
#         print(f"Workflows Completed On Time: {workflows_completed_on_time}")
#         print(f"Tasks Executed on Edge: {tasks_on_edge}")
#         print(f"Tasks Executed on Cloud: {tasks_on_cloud}")
#         print(f"Total Cold Starts: {cold_starts}")
#         # Re-enabled the print for Average Edge Utilization
#         print(f"Average Edge Utilization: {avg_util_percentage:.2f}%")
#         print("------------------------------------------") # Corrected from "                                                    "
#
#         with open(summary_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             if f.tell() == 0:
#                 writer.writerow(["Policy", "Workflows Submitted", "Workflows Completed", "Workflows On Time",
#                                  "Tasks on Edge", "Tasks on Cloud", "Cold Starts", "Avg Edge Utilization (%)"])
#             writer.writerow(
#                 [policy_name, total_workflows_submitted, total_workflows_completed, workflows_completed_on_time,
#                  tasks_on_edge, tasks_on_cloud, cold_starts, f"{avg_util_percentage:.2f}"])
#
#
#         with open(per_workflow_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             if f.tell() == 0:
#                 writer.writerow(["Policy", "Workflow ID", "Workflow Name", "Total Tasks", "Tasks On Time",
#                                  "Edge Tasks", "Cloud Tasks", "Cold Starts", "Avg Wait Time (ms)",
#                                  "Completed Within Deadline"])
#             for stat in workflow_stats:
#                 writer.writerow([policy_name, stat.workflow_id, stat.workflow_name, stat.total_tasks,
#                                  stat.tasks_completed_within_deadline, stat.tasks_executed_on_edge,
#                                  stat.tasks_executed_on_cloud, stat.cold_starts,
#                                  f"{stat.get_average_wait_time():.2f}",
#                                  stat.workflow_completed_within_deadline])
#
#         print(f"Results saved to {summary_file} and {per_workflow_file}")


import csv
import heapq
import os
import random
import time
from typing import Dict
from typing import List

from .cache_manager import CacheManager
from .config import (
    SIMULATION_DURATION, RANDOM_SEED, PREDICTION_INTERVAL,
    MIN_WORKFLOW_SUBMISSION_INTERVAL, MAX_WORKFLOW_SUBMISSION_INTERVAL,
    TOTAL_WORKFLOW_SUBMISSIONS
)
from .edge_network import EdgeNetwork
from .models import Task, Workflow, Cloud, WorkflowStats
from .parser import PegasusWorkflowParser
from .scheduler import Scheduler
from .results import plotting


class Simulator:
    def __init__(self, num_edge_nodes, cache_size, cold_start_penalty,
                 edge_latency, cloud_latency, workflow_filepaths,
                 adjacency_matrix, scheduling_policy, cache_sharing_policy,
                 public_cache_fraction=0.0):

        random.seed(RANDOM_SEED)
        self.current_time = 0.0
        self.event_queue = []
        self._event_id_counter = 0

        self.num_edge_nodes = num_edge_nodes
        self.cache_size = cache_size
        self.cold_start_penalty = cold_start_penalty
        self.edge_latency = edge_latency
        self.cloud_latency = cloud_latency
        self.workflow_filepaths = workflow_filepaths
        self.adjacency_matrix = adjacency_matrix
        self.scheduling_policy = scheduling_policy
        self.cache_sharing_policy = cache_sharing_policy
        self.public_cache_fraction = public_cache_fraction

        self.edge_network = EdgeNetwork(num_edge_nodes, edge_latency, adjacency_matrix)
        for node in self.edge_network.edge_nodes:
            node.capacity = 2 if self.cache_sharing_policy == "partial_public_private" else 1
            node.cache_size = self.cache_size

        self.cloud = Cloud(base_latency=self.cloud_latency)
        self.cache_manager = CacheManager(cache_size, num_edge_nodes, "LRU", cache_sharing_policy,
                                          public_cache_fraction)
        self.scheduler = Scheduler(self.edge_network, self.cloud, self.cache_manager, cold_start_penalty,
                                   scheduling_policy)
        self.parser = PegasusWorkflowParser()

        self.workflow_templates: Dict[str, Workflow] = {}
        self.workflow_stats: Dict[str, WorkflowStats] = {}

        self.global_total_workflows_completed = 0
        self.global_workflows_completed_on_time = 0
        self.global_tasks_on_edge = 0
        self.global_tasks_on_cloud = 0
        self.global_cold_starts = 0
        self.global_edge_utilization: Dict[int, List[int]] = {i: [] for i in range(self.num_edge_nodes)}
        self.last_utilization_record_time = 0.0

        self.edge_requests_count: Dict[int, int] = {i: 0 for i in range(self.num_edge_nodes)}

        self._load_workflow_templates()
        self._schedule_initial_events()

        print(
            f"Simulator initialized with scheduling policy: {self.scheduling_policy} and cache policy: {self.cache_sharing_policy}")

    def _add_event(self, time: float, event_type: str, *args):
        heapq.heappush(self.event_queue, (time, self._event_id_counter, event_type, args))
        self._event_id_counter += 1

    def _load_workflow_templates(self):
        for filepath in self.workflow_filepaths:
            workflow_template = self.parser.parse_workflow_template(filepath)
            if workflow_template:
                self.workflow_templates[workflow_template.name] = workflow_template
            else:
                print(f"Warning: Failed to load workflow template from {filepath}")

    def _schedule_initial_events(self):
        available_workflow_names = list(self.workflow_templates.keys())
        if not available_workflow_names:
            print("No workflow templates loaded. Cannot schedule submission events.")
            return

        current_event_time = 0.0
        for i in range(TOTAL_WORKFLOW_SUBMISSIONS):
            workflow_name_to_submit = random.choice(available_workflow_names)
            submission_interval = random.uniform(MIN_WORKFLOW_SUBMISSION_INTERVAL, MAX_WORKFLOW_SUBMISSION_INTERVAL)
            current_event_time += submission_interval
            self._add_event(current_event_time, "workflow_submission", workflow_name_to_submit)

        print(f"Generated {TOTAL_WORKFLOW_SUBMISSIONS} workflow submission events.")
        self._add_event(PREDICTION_INTERVAL, "predictive_prefetch")
        self._add_event(SIMULATION_DURATION + 1, "simulation_end")

    def _process_workflow_submission(self, workflow_name: str):
        workflow_template = self.workflow_templates.get(workflow_name)
        if not workflow_template:
            print(f"Error: Workflow template '{workflow_name}' not found for submission.")
            return

        workflow_instance_id = f"{workflow_template.id}_inst_{int(self.current_time)}_{random.randint(0, 9999)}"

        workflow_instance = workflow_template.create_instance(
            new_id=workflow_instance_id,
            current_simulation_time=self.current_time
        )

        self.workflow_stats[workflow_instance.id] = WorkflowStats(workflow_instance.id, workflow_instance.name,
                                                                  workflow_instance.total_tasks)

        self.workflow_stats[workflow_instance.id].workflow_instance = workflow_instance

        for task in workflow_instance.get_ready_tasks():
            task.ready_time = self.current_time
            self.scheduler.add_ready_task(task)

    def _process_task_completion(self, task: Task):
        workflow_instance = task.workflow_instance
        if not workflow_instance:
            print(f"Error: Workflow instance not found for task {task.id}.")
            return

        self.global_cold_starts += 1 if task.cold_start_penalty > 0 else 0
        if task.assigned_node_type == "edge":
            self.global_tasks_on_edge += 1
            if task.assigned_node and hasattr(task.assigned_node, 'id'):
                self.edge_requests_count[task.assigned_node.id] += 1
        elif task.assigned_node_type == "cloud":
            self.global_tasks_on_cloud += 1

        self.workflow_stats[workflow_instance.id].add_task_execution_detail(
            task.assigned_node_type,
            task.end_time <= task.deadline,
            task.wait_time,
            task.cold_start_penalty
        )

        workflow_instance.mark_task_completed(task.id)

        for next_task in workflow_instance.get_successors(task.id):
            all_deps_met = all(dep_id in workflow_instance.completed_tasks for dep_id in next_task.dependencies)
            if all_deps_met and next_task.start_time is None:
                next_task.ready_time = self.current_time
                self.scheduler.add_ready_task(next_task)

        if workflow_instance.is_completed():
            self.global_total_workflows_completed += 1
            workflow_instance.end_time = self.current_time
            wf_stats_obj = self.workflow_stats.get(workflow_instance.id)
            if wf_stats_obj:
                wf_stats_obj.workflow_completed_within_deadline = workflow_instance.check_if_completed_within_deadline(
                    self.current_time)
                if wf_stats_obj.workflow_completed_within_deadline:
                    self.global_workflows_completed_on_time += 1

    def _process_predictive_prefetch(self):
        self.scheduler.predictive_prefetch(self.workflow_stats, self.current_time)
        self._add_event(self.current_time + PREDICTION_INTERVAL, "predictive_prefetch")

    def _record_edge_utilization(self):
        for node in self.edge_network.get_all_edge_nodes():
            is_busy = 1 if node.is_busy(self.current_time) else 0
            self.global_edge_utilization[node.id].append(is_busy)

    def run_simulation(self):
        start_time_real = time.time()
        print(f"Starting simulation for {SIMULATION_DURATION / 1000 / 60:.2f} minutes...")

        all_function_ids = list(self.parser.functions_catalog.keys())
        self.scheduler.initial_prefetch(all_function_ids)

        self.last_utilization_record_time = 0.0

        while self.event_queue and self.current_time <= SIMULATION_DURATION:
            event_time, _, event_type, event_args = heapq.heappop(self.event_queue)

            if event_time > self.current_time:
                self.current_time = event_time
                while self.last_utilization_record_time + PREDICTION_INTERVAL <= self.current_time:
                    self._record_edge_utilization()
                    self.last_utilization_record_time += PREDICTION_INTERVAL

            if self.current_time > SIMULATION_DURATION:
                break

            if event_type == "workflow_submission":
                self._process_workflow_submission(*event_args)
            elif event_type == "task_completion":
                self._process_task_completion(*event_args)
            elif event_type == "predictive_prefetch":
                self._process_predictive_prefetch()
            elif event_type == "simulation_end":
                print(
                    f"Simulation ended at {self.current_time / 1000:.2f} seconds ({self.current_time / 1000 / 60:.2f} minutes).")
                break

            scheduled_tasks = self.scheduler.schedule_tasks(self.current_time, self.workflow_stats)
            for task, assigned_node, assigned_node_type, cold_start_penalty, wait_time in scheduled_tasks:
                task.start_time = self.current_time + wait_time
                task.end_time = task.start_time + task.runtime
                task.assigned_node = assigned_node
                task.assigned_node_type = assigned_node_type
                task.cold_start_penalty = cold_start_penalty
                task.wait_time = wait_time

                self._add_event(task.end_time, "task_completion", task)

        while self.last_utilization_record_time + PREDICTION_INTERVAL <= self.current_time:
            self._record_edge_utilization()
            self.last_utilization_record_time += PREDICTION_INTERVAL

        end_time_real = time.time()
        print(f"Total simulation wall time: {end_time_real - start_time_real:.2f} seconds.")

        self._save_results(
            policy_name=f"{self.scheduling_policy}_{self.cache_sharing_policy}",
            total_workflows_submitted=TOTAL_WORKFLOW_SUBMISSIONS,
            total_workflows_completed=self.global_total_workflows_completed,
            workflows_completed_on_time=self.global_workflows_completed_on_time,
            tasks_on_edge=self.global_tasks_on_edge,
            tasks_on_cloud=self.global_tasks_on_cloud,
            cold_starts=self.global_cold_starts,
            edge_utilization_data=self.global_edge_utilization,
            workflow_stats=list(self.workflow_stats.values()),
            edge_request_counts=self.edge_requests_count
        )

    def _save_results(self, policy_name: str, total_workflows_submitted: int,
                      total_workflows_completed: int, workflows_completed_on_time: int,
                      tasks_on_edge: int, tasks_on_cloud: int, cold_starts: int,
                      edge_utilization_data: Dict[int, List[int]],
                      workflow_stats: List[WorkflowStats],
                      edge_request_counts: Dict[int, int]):
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        summary_file = os.path.join(results_dir, "simulation_summary.csv")
        per_workflow_file = os.path.join(results_dir, "per_workflow_stats.csv")

        total_busy_intervals = sum(sum(util) for util in edge_utilization_data.values())
        total_intervals_recorded = sum(len(util) for util in edge_utilization_data.values())

        avg_util_percentage = (total_busy_intervals / total_intervals_recorded) * 100 if total_intervals_recorded > 0 else 0.0

        print(f"\n=== Simulation Results ({policy_name}) ===")
        print(f"Total Workflows Submitted: {total_workflows_submitted}")
        print(f"Total Workflows Completed: {total_workflows_completed}")
        print(f"Workflows Completed On Time: {workflows_completed_on_time}")
        print(f"Tasks Executed on Edge: {tasks_on_edge}")
        print(f"Tasks Executed on Cloud: {tasks_on_cloud}")
        print(f"Total Cold Starts: {cold_starts}")
        print(f"Average Edge Utilization: {avg_util_percentage:.2f}%")
        print("------------------------------------------")

        with open(summary_file, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Policy", "Workflows Submitted", "Workflows Completed", "Workflows On Time",
                                 "Tasks on Edge", "Tasks on Cloud", "Cold Starts", "Avg Edge Utilization (%)"])
            writer.writerow(
                [policy_name, total_workflows_submitted, total_workflows_completed, workflows_completed_on_time,
                 tasks_on_edge, tasks_on_cloud, cold_starts, f"{avg_util_percentage:.2f}"])

        with open(per_workflow_file, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Policy", "Workflow ID", "Workflow Name", "Total Tasks", "Tasks On Time",
                                 "Edge Tasks", "Cloud Tasks", "Cold Starts", "Avg Wait Time (ms)",
                                 "Completed Within Deadline"])
            for stat in workflow_stats:
                writer.writerow([policy_name, stat.workflow_id, stat.workflow_name, stat.total_tasks,
                                 stat.tasks_completed_within_deadline, stat.tasks_executed_on_edge,
                                 stat.tasks_executed_on_cloud, stat.cold_starts,
                                 f"{stat.get_average_wait_time():.2f}",
                                 stat.workflow_completed_within_deadline])

        # Save edge request heatmap
        plotting.plot_edge_request_heatmap(edge_request_counts, self.num_edge_nodes, policy_name)
        print(f"Results saved to {summary_file} and {per_workflow_file}")
