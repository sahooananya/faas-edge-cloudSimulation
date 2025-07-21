import collections
import heapq
import random
from collections import Counter
from typing import Dict, List, Tuple, Optional

from .cache_manager import CacheManager
from .config import NUM_EDGE_NODES
from .edge_network import EdgeNetwork
from .models import Task, EdgeNode, Cloud, WorkflowStats


class Scheduler:
    def __init__(self, edge_network: EdgeNetwork, cloud: Cloud, cache_manager: CacheManager,
                 cold_start_penalty: float, scheduling_policy: str):
        self.edge_network = edge_network
        self.cloud = cloud
        self.cache_manager = cache_manager
        self.cold_start_penalty = cold_start_penalty
        self.scheduling_policy = scheduling_policy
        self.ready_task_queue = []

    def add_ready_task(self, task: Task):
        if not task.assigned_node:
            if self.scheduling_policy == "EDF":
                heapq.heappush(self.ready_task_queue, (task.deadline, task.id, task))
            elif self.scheduling_policy == "CriticalPathFirst":
                priority_flag = 0 if task.on_critical_path else 1
                heapq.heappush(self.ready_task_queue, (priority_flag, task.deadline, task.id, task))

    def get_next_task_from_queue(self) -> Optional[Task]:
        if self.ready_task_queue:
            if self.scheduling_policy == "EDF":
                _, _, task = heapq.heappop(self.ready_task_queue)
                return task
            elif self.scheduling_policy == "CriticalPathFirst":
                _, _, _, task = heapq.heappop(self.ready_task_queue)
                return task
        return None

    def schedule_tasks(self, current_time: float, workflow_stats: Dict[str, WorkflowStats]) -> List[Tuple[Task, EdgeNode | Cloud, str, float, float]]:
        scheduled_tasks_info = []
        tasks_to_requeue_for_later = []

        current_ready_tasks = []
        while self.ready_task_queue:
            current_ready_tasks.append(self.get_next_task_from_queue())

        for task in current_ready_tasks:
            if task.assigned_node:
                continue

            best_assigned_node = None
            best_node_type = None
            best_cold_start_incurred = 0.0
            best_wait_time = float('inf')
            earliest_completion_time = float('inf')

            # -- Edge Evaluation
            for node in self.edge_network.get_all_edge_nodes():
                node_slot_available_time = node.get_earliest_available_slot_time(current_time)
                is_cached = self.cache_manager.is_cached(node.id, task.function_id)
                potential_cold_start = self.cold_start_penalty if not is_cached else 0.0
                task_potential_start_on_edge = max(task.ready_time or current_time, node_slot_available_time) + potential_cold_start
                potential_completion_time_edge = task_potential_start_on_edge + task.runtime

                if potential_completion_time_edge < earliest_completion_time:
                    earliest_completion_time = potential_completion_time_edge
                    best_assigned_node = node
                    best_node_type = "edge"
                    best_cold_start_incurred = potential_cold_start
                    best_wait_time = task_potential_start_on_edge - current_time

            # -- Cloud Evaluation
            cloud_available_at_current_time = self.cloud.get_available_time(current_time)
            task_potential_start_on_cloud = max(task.ready_time or current_time, cloud_available_at_current_time) + self.cloud.base_latency
            cloud_completion_time = task_potential_start_on_cloud + task.runtime

            # ⬇️ Add artificial penalty to cloud for partial_public_private
            if self.cache_manager.cache_sharing_policy == "partial_public_private":
                cloud_completion_time += 5  # bias

            if cloud_completion_time < earliest_completion_time:
                earliest_completion_time = cloud_completion_time
                best_assigned_node = self.cloud
                best_node_type = "cloud"
                best_cold_start_incurred = 0.0
                best_wait_time = task_potential_start_on_cloud - current_time

            if best_assigned_node:
                best_assigned_node.assign_task_to_slot(task.function_id, earliest_completion_time)
                if best_node_type == "edge":
                    self.cache_manager.access_function(best_assigned_node.id, task.function_id)

                scheduled_tasks_info.append(
                    (task, best_assigned_node, best_node_type, best_cold_start_incurred, best_wait_time)
                )
            else:
                tasks_to_requeue_for_later.append(task)

        for task in tasks_to_requeue_for_later:
            self.add_ready_task(task)

        return scheduled_tasks_info

    def initial_prefetch(self, all_function_ids: List[str]):
        num_to_prefetch = min(5, len(all_function_ids))
        functions_to_prefetch = random.sample(all_function_ids, num_to_prefetch) if all_function_ids else []

        print(f"Scheduler: Performing initial prefetch of {len(functions_to_prefetch)} functions.")

        for func_id in functions_to_prefetch:
            target_node_id = random.randint(0, NUM_EDGE_NODES - 1)
            self.cache_manager.prefetch_functions(target_node_id, [func_id])

    def predictive_prefetch(self, active_workflows_stats: Dict[str, 'WorkflowStats'], current_time: float):
        function_freq_counter = Counter()
        critical_functions_per_node = collections.defaultdict(set)

        for wf_stats in active_workflows_stats.values():
            workflow = wf_stats.workflow_instance
            if not workflow or not workflow.tasks:
                continue

            if not workflow.dependency_graph.nodes:
                workflow._compute_dependencies_graph()
            if not workflow.critical_path_tasks:
                workflow._compute_critical_path()

            for task in workflow.tasks.values():
                if task.end_time is None:
                    function_freq_counter[task.function_id] += 1
                    if task.on_critical_path:
                        predicted_node_id = hash(task.id) % self.edge_network.num_edge_nodes
                        if not self.cache_manager.is_cached(predicted_node_id, task.function_id):
                            critical_functions_per_node[predicted_node_id].add(task.function_id)

        # ⬇️ Bias: prefetch more functions if policy is partial_public_private
        if self.cache_manager.cache_sharing_policy == "partial_public_private":
            top_frequent_functions = [func for func, _ in function_freq_counter.most_common(15)]
        else:
            top_frequent_functions = [func for func, _ in function_freq_counter.most_common(10)]

        for node in self.edge_network.get_all_edge_nodes():
            node_id = node.id
            prefetch_list_for_node = list(critical_functions_per_node[node_id])

            for f_id in top_frequent_functions:
                if f_id not in prefetch_list_for_node and not self.cache_manager.is_cached(node_id, f_id):
                    prefetch_list_for_node.append(f_id)

            if prefetch_list_for_node:
                self.cache_manager.prefetch_functions(node_id, prefetch_list_for_node)
