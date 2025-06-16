import heapq
from typing import Dict, Set, List, Tuple, Optional
import collections
from collections import Counter
import random
from .models import Task, EdgeNode, Cloud, Workflow, WorkflowStats  # Import WorkflowStats
from .cache_manager import CacheManager
from .edge_network import EdgeNetwork
from .config import COLD_START_PENALTY, NUM_EDGE_NODES


class Scheduler:
    def __init__(self, edge_network: EdgeNetwork, cloud: Cloud, cache_manager: CacheManager,
                 cold_start_penalty: float, scheduling_policy: str):
        self.edge_network = edge_network
        self.cloud = cloud
        self.cache_manager = cache_manager
        self.cold_start_penalty = cold_start_penalty
        self.scheduling_policy = scheduling_policy
        self.ready_task_queue = []  # Min-heap for (priority, task_id, task_obj)

    def add_ready_task(self, task: Task):
        """
        Adds a task to the ready queue based on the current scheduling policy.
        Ensures task is not already assigned and its dependencies are met.
        """
        # A task is ready if its dependencies are met and it hasn't been assigned yet.
        # The simulator calls this after dependencies are met.
        if not task.assigned_node:
            if self.scheduling_policy == "EDF":
                heapq.heappush(self.ready_task_queue, (task.deadline, task.id, task))
            elif self.scheduling_policy == "CriticalPathFirst":
                priority_flag = 0 if task.on_critical_path else 1
                heapq.heappush(self.ready_task_queue, (priority_flag, task.deadline, task.id, task))

    def get_next_task_from_queue(self) -> Optional[Task]:
        """Retrieves the next task to be scheduled from the ready queue and removes it."""
        if self.ready_task_queue:
            if self.scheduling_policy == "EDF":
                _, _, task = heapq.heappop(self.ready_task_queue)
                return task
            elif self.scheduling_policy == "CriticalPathFirst":
                _, _, _, task = heapq.heappop(self.ready_task_queue)
                return task
        return None

    def schedule_tasks(self, current_time: float, workflow_stats: Dict[str, WorkflowStats]):  # Changed type hint
        """
        Attempts to schedule ready tasks on appropriate nodes (edge or cloud).
        Returns a list of successfully scheduled tasks with their details.
        """
        scheduled_tasks_info: List[Tuple[Task, EdgeNode | Cloud, str, float, float]] = []
        tasks_to_requeue_for_later = []  # Tasks that couldn't be scheduled in this iteration

        # To avoid modifying ready_task_queue while iterating, drain it first.
        current_ready_tasks = []
        while self.ready_task_queue:
            current_ready_tasks.append(self.get_next_task_from_queue())

        for task in current_ready_tasks:
            if task.assigned_node:  # Skip if already assigned (defensive check)
                continue

            best_assigned_node = None
            best_node_type = None
            best_cold_start_incurred = 0.0
            best_wait_time = 0.0  # Time task spends waiting for resource + cold start
            earliest_completion_time = float('inf')

            # --- 1. Evaluate Edge Nodes ---
            for node in self.edge_network.get_all_edge_nodes():
                # Get the earliest time a slot on this node becomes free for a new task
                node_slot_available_time = node.get_earliest_available_slot_time(current_time)

                # If the node has no available slots (shouldn't be None, but defensive)
                if node_slot_available_time is None:
                    continue

                    # Check cache status for this function on this specific edge node
                is_cached = self.cache_manager.is_cached(node.id, task.function_id)
                potential_cold_start = self.cold_start_penalty if not is_cached else 0.0

                # Calculate potential start time on this edge node
                # It's the later of when the task is ready and when a slot is available on the node, plus cold start
                task_potential_start_on_edge = max(task.ready_time if task.ready_time is not None else current_time,
                                                   node_slot_available_time) + potential_cold_start

                potential_completion_time_edge = task_potential_start_on_edge + task.runtime

                if potential_completion_time_edge < earliest_completion_time:
                    earliest_completion_time = potential_completion_time_edge
                    best_assigned_node = node
                    best_node_type = "edge"
                    best_cold_start_incurred = potential_cold_start
                    # Wait time here is total delay from current_time until actual execution starts
                    best_wait_time = task_potential_start_on_edge - current_time

            # --- 2. Evaluate Cloud ---
            # Cloud is assumed to have high capacity, so it's ready as soon as the task is ready
            # + base latency to send task to cloud.
            cloud_available_time = self.cloud.get_available_time(current_time)
            task_potential_start_on_cloud = max(task.ready_time if task.ready_time is not None else current_time,
                                                cloud_available_time) + self.cloud.base_latency
            cloud_completion_time = task_potential_start_on_cloud + task.runtime

            if cloud_completion_time < earliest_completion_time:
                earliest_completion_time = cloud_completion_time
                best_assigned_node = self.cloud
                best_node_type = "cloud"
                best_cold_start_incurred = 0.0  # No cold start for cloud in this model
                best_wait_time = task_potential_start_on_cloud - current_time

            # --- Assign Task to the Best Node ---
            if best_assigned_node:
                # If a node was chosen, "assign" the task to it.
                # The actual task object state (start_time, end_time, assigned_node) will be set in simulator.py
                # This call updates the node's internal state (e.g., available_time for Cloud, active_slots for EdgeNode)
                if best_node_type == "edge":
                    actual_start_time, incurred_cold_start = best_assigned_node.assign_task(
                        task.function_id, task.runtime, self.cold_start_penalty, current_time
                    )
                    # Update cache manager's LRU for this access
                    self.cache_manager.access_function(best_assigned_node.id, task.function_id)
                else:  # cloud
                    actual_start_time, incurred_cold_start = best_assigned_node.assign_task(
                        task.function_id, task.runtime, current_time  # Cold start penalty is typically 0 for cloud
                    )

                # Append details to scheduled_tasks_info
                scheduled_tasks_info.append(
                    (task, best_assigned_node, best_node_type, incurred_cold_start, actual_start_time - current_time))
            else:
                # If for some reason no node was found (e.g., all busy and cloud also too slow, or some logic error)
                # Re-queue the task for next iteration.
                tasks_to_requeue_for_later.append(task)

        # Re-add tasks that could not be scheduled to the queue, maintaining their priority
        for task in tasks_to_requeue_for_later:
            self.add_ready_task(task)

        return scheduled_tasks_info

    def initial_prefetch(self, all_function_ids: List[str]):
        """
        Performs initial prefetch of a fixed number of available functions
        to random edge caches.
        The `all_function_ids` is a list of all unique function IDs available in the system.
        """
        # Ensure we don't try to prefetch more functions than available or than cache can hold.
        num_to_prefetch = min(5, len(all_function_ids))  # Prefetch max 5 functions, or fewer if not enough exist

        # Shuffle a copy of the list to pick 'random' functions for initial prefetch
        functions_to_prefetch = random.sample(all_function_ids, num_to_prefetch) if all_function_ids else []

        print(f"Scheduler: Performing initial prefetch of {len(functions_to_prefetch)} functions.")

        # Distribute these functions across edge nodes randomly
        for func_id in functions_to_prefetch:
            target_node_id = random.randint(0, NUM_EDGE_NODES - 1)
            # Pass func_id as a list, as prefetch_functions expects a list of strings
            self.cache_manager.prefetch_functions(target_node_id, [func_id])

    def predictive_prefetch(self, active_workflows_stats: Dict[str, 'WorkflowStats'], current_time: float):
        """
        Predictively load functions to edge caches based on critical path importance and usage frequency.
        `active_workflows_stats` is a dictionary of WorkflowStats objects, which contain Workflow instances.
        """
        function_freq_counter = Counter()
        critical_functions_per_node: Dict[int, Set[str]] = collections.defaultdict(set)

        for wf_stats in active_workflows_stats.values():
            workflow = wf_stats.workflow_instance  # Get the actual Workflow object
            if not workflow or not workflow.tasks:
                continue

            # Ensure dependency graph and critical path are computed for this workflow instance
            # (they should be from template, but a check or recompute doesn't hurt)
            if not workflow.dependency_graph.nodes:
                workflow._compute_dependencies_graph()
            if not workflow.critical_path_tasks:
                workflow._compute_critical_path()

            for task in workflow.tasks.values():
                if task.end_time is None:  # Only consider tasks not yet completed
                    function_freq_counter[task.function_id] += 1

                    if task.on_critical_path:
                        # Heuristic: Predict which node a critical task might run on.
                        # Simple: hash task ID to a node ID. More advanced: use historical data or current load.
                        predicted_node_id = hash(task.id) % self.edge_network.num_edge_nodes

                        # Only prefetch if not already cached on that node
                        if not self.cache_manager.is_cached(predicted_node_id, task.function_id):
                            critical_functions_per_node[predicted_node_id].add(task.function_id)

        # Prioritize top N most frequent functions across all active workflows
        top_frequent_functions = [func for func, _ in function_freq_counter.most_common(10)]  # Top 10 most frequent

        # Now populate caches for critical functions and top frequent ones
        for node in self.edge_network.get_all_edge_nodes():
            node_id = node.id
            prefetch_list_for_node = []

            # Add functions predicted to be critical for this specific node
            prefetch_list_for_node.extend(list(critical_functions_per_node[node_id]))

            # Add globally popular functions that are not already in the critical list for this node's prefetch
            for f_id in top_frequent_functions:
                if f_id not in prefetch_list_for_node and not self.cache_manager.is_cached(node_id, f_id):
                    prefetch_list_for_node.append(f_id)

            # Perform prefetch using the CacheManager
            if prefetch_list_for_node:
                self.cache_manager.prefetch_functions(node_id, prefetch_list_for_node)
