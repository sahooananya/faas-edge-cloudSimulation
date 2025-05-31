# src/scheduler.py

import heapq
from typing import Dict, Any, Set  # <--- ADD Dict and Set here
import collections  # <--- ADD collections for defaultdict

from .models import Task, EdgeNode, Cloud, Workflow
from .cache_manager import CacheManager
from .edge_network import EdgeNetwork
from .config import COLD_START_PENALTY, CLOUD_TO_EDGE_LATENCY, EDGE_TO_EDGE_LATENCY, EDGE_NODES_CONNECTED_TO_CLOUD


class Scheduler:
    def __init__(self, edge_network: EdgeNetwork, cloud: Cloud, cache_manager: CacheManager, cold_start_penalty: float,
                 scheduling_policy: str):
        self.edge_network = edge_network
        self.cloud = cloud
        self.cache_manager = cache_manager
        self.cold_start_penalty = cold_start_penalty

        self.scheduling_policy = scheduling_policy
        self.edge_node_available_time = {node.id: 0.0 for node in self.edge_network.get_all_edge_nodes()}
        self.ready_task_queue = []

    def add_ready_task(self, task: Task):
        if self.scheduling_policy == "EDF":
            heapq.heappush(self.ready_task_queue, (task.deadline, task.id, task))
        elif self.scheduling_policy == "CriticalPathFirst":
            priority_flag = 0 if task.on_critical_path else 1
            heapq.heappush(self.ready_task_queue, (priority_flag, task.deadline, task.id, task))

    def get_next_task_to_schedule(self, current_time: float):
        if not self.ready_task_queue:
            return None
        return self.ready_task_queue[0]

    def find_best_execution_option(self, task: Task, current_time: float):
        best_node = None
        best_node_type = None
        min_completion_time = float('inf')
        cold_start_incurred = 0.0
        calculated_wait_time = 0.0

        primary_edge_id = hash(task.id) % self.edge_network.num_edge_nodes
        primary_edge = self.edge_network.get_edge_node_by_id(primary_edge_id)

        edges_to_evaluate = []
        if primary_edge:
            edges_to_evaluate.append(primary_edge)
            for neighbor in self.edge_network.get_neighbors(primary_edge.id):
                if neighbor.id != primary_edge.id and neighbor not in edges_to_evaluate:
                    edges_to_evaluate.append(neighbor)

        for node in self.edge_network.get_all_edge_nodes():
            if node not in edges_to_evaluate:
                edges_to_evaluate.append(node)

        for edge_node in edges_to_evaluate:
            current_node_available_time = self.edge_node_available_time[edge_node.id]

            cold_start_cost = 0.0
            if not self.cache_manager.is_cached(edge_node.id, task.function_id):
                cold_start_cost = self.cold_start_penalty

            estimated_start_time_at_node = max(current_time,
                                               current_node_available_time) + self.edge_network.base_latency + cold_start_cost
            estimated_completion_time = estimated_start_time_at_node + task.runtime

            if estimated_completion_time < min_completion_time:
                min_completion_time = estimated_completion_time
                best_node = edge_node
                best_node_type = "edge"
                cold_start_incurred = cold_start_cost
                calculated_wait_time = max(0.0, current_node_available_time - current_time)

        cloud_est_completion_time = current_time + self.cloud.latency + task.runtime

        if best_node is None or cloud_est_completion_time < min_completion_time:
            min_completion_time = cloud_est_completion_time
            best_node = self.cloud
            best_node_type = "cloud"
            cold_start_incurred = 0.0
            calculated_wait_time = 0.0

        if best_node_type == "edge":
            self.cache_manager.access_function(best_node.id, task.function_id)

        return best_node, best_node_type, cold_start_incurred, calculated_wait_time, min_completion_time

    def assign_task(self, task: Task, assigned_node, node_type, cold_start_cost, wait_time):
        task.assigned_node = assigned_node
        task.assigned_node_type = node_type
        task.cold_start_penalty = cold_start_cost
        task.wait_time = wait_time

    def update_node_busy_time(self, node, end_time, node_type):
        if node_type == "edge":
            self.edge_node_available_time[node.id] = max(self.edge_node_available_time[node.id], end_time)

    def trigger_predictive_cache_load(self, active_workflows: Dict[str, Workflow], current_time: float):
        """
        Implements a more robust predictive cache loading strategy based on workflow DAG.
        It identifies functions for tasks that are "imminent" (few dependencies left)
        and prioritizes those on the critical path.
        """
        functions_to_prefetch: Dict[int, Set[str]] = collections.defaultdict(set)  # Corrected type hint for Set

        for workflow_id, workflow in active_workflows.items():
            if not workflow.dependency_graph:  # Ensure graph is built (should be from parser)
                workflow._compute_dependencies_graph()

            for task_id, task in workflow.tasks.items():
                if task.end_time is None:  # Only consider uncompleted tasks
                    remaining_dependencies = len(task.dependencies)

                    is_imminent_task = (remaining_dependencies == 0 or remaining_dependencies == 1)
                    is_critical_imminent = task.on_critical_path and remaining_dependencies <= 2

                    if is_imminent_task or is_critical_imminent:
                        predicted_node_id = hash(task.id) % self.edge_network.num_edge_nodes
                        predicted_edge_node = self.edge_network.get_edge_node_by_id(predicted_node_id)

                        if predicted_edge_node:  # Only consider edge-bound tasks for prefetching
                            functions_to_prefetch[predicted_edge_node.id].add(task.function_id)

        for node_id, func_ids_set in functions_to_prefetch.items():
            func_ids_list = list(func_ids_set)
            self.cache_manager.predict_and_load(node_id, func_ids_list)