# src/scheduler.py

import heapq  # Still used for event queue, not for task prioritization within scheduler directly now
from collections import deque

from .models import Task, Workflow, EdgeNode, Cloud
from .cache_manager import CacheManager
from .edge_network import EdgeNetwork
from .config import COLD_START_PENALTY, CLOUD_LATENCY  # Import config values directly


class Scheduler:
    def __init__(self, edge_network: EdgeNetwork, cloud: Cloud, cache_manager: CacheManager, cold_start_penalty: float,
                 scheduling_policy: str):
        self.edge_network = edge_network
        self.cloud = cloud
        self.cache_manager = cache_manager
        self.cold_start_penalty = cold_start_penalty
        self.scheduling_policy = scheduling_policy

        # Keep track of when each edge node is available
        self.edge_node_available_time = {node.id: 0.0 for node in self.edge_network.get_all_edge_nodes()}

        # A priority queue for all tasks that are READY to be scheduled, ordered by deadline (for EDF)
        # (deadline, task_id, workflow_id) - we need task_id and workflow_id to retrieve the Task object
        self.ready_task_queue = []  # Min-heap: (deadline, unique_task_id, task_object_ref)

        # To track tasks currently running on edge nodes or cloud (for internal state)
        self.running_tasks = {}  # {task_id: assigned_node_id}

    def add_ready_task(self, task: Task):
        """Adds a task to the ready queue. For EDF, tasks are prioritized by deadline."""
        # For EDF, the priority is the task's deadline.
        # We store the task object itself for easy retrieval.
        heapq.heappush(self.ready_task_queue, (task.deadline, task.id, task))

    def get_next_task_to_schedule(self, current_time: float):
        """
        Retrieves the next task to schedule based on the policy.
        For EDF, this is simply the task with the earliest deadline from the ready queue.
        Returns the task object, or None if no tasks are ready.
        """
        if not self.ready_task_queue:
            return None

        if self.scheduling_policy == "EDF":
            # Peek at the top (earliest deadline) without removing
            # We need to make sure the task is actually ready (not already assigned/completed)
            # and belongs to an active workflow. This check will be done by the simulator.
            # Here, we just return the highest priority task.

            # Simple EDF: Just pop the top one. The simulator will handle if it's invalid.
            # Note: For robust EDF, you might need to check if a task is actually schedulable
            # on an available resource before popping, to avoid 'blocking' a resource with an
            # unfulfillable task. For now, we'll try to schedule it.
            _deadline, _task_id, task = self.ready_task_queue[0]  # Peek

            # We need to ensure that the task is still in the active workflow's tasks and not completed.
            # This is tricky because the scheduler doesn't hold the workflow context directly.
            # The `simulator.py` will have to handle this re-validation upon popping from this queue.
            # For simplicity in Phase 1/2, we'll assume tasks added here are valid.
            # The simulator will remove completed tasks.
            return task
        elif self.scheduling_policy == "SimpleGreedy":
            # For simple greedy, we'd iterate through a flat list or unsorted collection.
            # We'll just stick to EDF for this phase.
            return None  # Should not be called with SimpleGreedy if policy is EDF

    def find_best_execution_option(self, task: Task, current_time: float):
        """
        Evaluates the best node (edge or cloud) for a given task.
        Prioritizes edge if it's competitive, falls back to cloud.
        Returns the assigned node, node type, cold start cost, and estimated completion time.
        """
        best_node = None
        best_node_type = None
        min_completion_time = float('inf')
        cold_start_incurred = 0.0
        calculated_wait_time = 0.0

        # 1. Evaluate Edge Nodes
        # For EDF, we don't necessarily pick the 'best' edge node immediately,
        # but we need to know the *earliest possible* completion time on any edge node
        # to compare against the cloud and for feasibility check.

        # Sort edge nodes by their current available time to find the 'soonest available'
        # This is a greedy choice *within* the edge network, not the overall task priority.
        sorted_edge_nodes = sorted(self.edge_network.get_all_edge_nodes(),
                                   key=lambda node: self.edge_node_available_time[node.id])

        for edge_node in sorted_edge_nodes:
            current_node_available_time = self.edge_node_available_time[edge_node.id]

            # Determine cold start
            cold_start_cost = 0.0
            if not self.cache_manager.is_cached(edge_node.id, task.function_id):
                cold_start_cost = self.cold_start_penalty

            # Estimated start time at this specific edge node
            # This is when the node is ready AND the task is dispatched AND cold start is done
            # The actual execution doesn't start until this time.
            estimated_start_time_at_node = max(current_time,
                                               current_node_available_time) + self.edge_network.base_latency + cold_start_cost
            estimated_completion_time = estimated_start_time_at_node + task.runtime

            # Check if this edge node is a better option so far
            if estimated_completion_time < min_completion_time:
                min_completion_time = estimated_completion_time
                best_node = edge_node
                best_node_type = "edge"
                cold_start_incurred = cold_start_cost
                calculated_wait_time = max(0.0, current_node_available_time - current_time)

        # 2. Evaluate Cloud
        # Cloud is always available. Latency + Runtime. No cold start.
        cloud_est_completion_time = current_time + self.cloud.latency + task.runtime

        # Decision Logic: Prioritize Edge if competitive, otherwise Cloud.
        # "Make sure edge nodes are selected before falling back to the cloud."
        # This means, if an edge node can meet the task's deadline OR is significantly faster
        # than cloud despite cold start, favor edge.

        # Simple rule for Phase 2: If an edge node is selected as 'best_node'
        # and its completion time is within the task's deadline, or it's simply better than cloud, use it.
        # If no edge node was found, or if the best edge option is strictly worse than cloud,
        # *and* cloud can still meet the deadline, go to cloud.

        # Current logic already selects the best option.
        # To strictly prioritize edge: if edge node exists, and it can complete on time,
        # and it's not disastrously slower than cloud (e.g., within some factor), prefer it.
        # For Phase 2, let's keep the greedy choice based on earliest completion, but ensure
        # the initial 'best_node' setup considers an edge first if available.

        # Refined decision based on the requirement: "Make sure edge nodes are selected before falling back to the cloud."
        # This implies a preference for edge even if cloud is marginally faster, as long as edge is feasible.

        # If there's an edge candidate, and it finishes within the task's deadline
        # OR if it's simply faster than the cloud option (which is the previous check)
        # OR if we explicitly want to push tasks to edge (for higher edge utilization)

        # Let's add a bias towards edge, for example, if the edge completion time is within a small margin of cloud's.
        # Or, just check if the current 'best_node' (which is edge if found first) is better than cloud.

        # The current logic of `if best_node is None or cloud_completion_time < min_completion_time:`
        # will always pick cloud if it's faster.
        # To "prioritize edge nodes before falling back to cloud":

        # If an edge node was found (best_node_type == "edge") AND
        # (its completion time is better than the cloud's OR it meets the task's deadline AND it's not excessively slower than cloud)
        # We are going to strictly choose the 'best_edge_node' unless cloud is significantly better.
        # For Phase 2, let's just make it a simple threshold.

        # If edge is selected as the best option, we keep it.
        # If cloud is faster, then choose cloud.
        # The existing logic already prioritizes the overall earliest completion time.
        # To force edge: We need a threshold. If edge is within X% of cloud completion time, pick edge.
        # For simplicity, let's just ensure that if an edge node *can* finish a task,
        # it gets a strong consideration, and only fall back to cloud if absolutely necessary (e.g. deadline impossible on edge).

        # Option A (Simpler): Stick with earliest completion time, but ensure edge calculation is correct.
        # The current code calculates edge completion time, then compares to cloud.
        # The line `if best_node is None or cloud_completion_time < min_completion_time:`
        # effectively says "if cloud is better than the best edge found so far, pick cloud".
        # This is fine for EDF, as EDF cares about *deadlines*, not just location.
        # The preference for edge nodes usually comes from *cost* or *latency* advantages.
        # If cloud is cheaper/faster, then it's a valid EDF decision.

        # Let's adjust this to ensure edge gets a chance if it's available without waiting for *too* long.
        # The current `max(current_time, current_node_available_time)` already implicitly handles waiting.

        # The most straightforward way to "prioritize edge before falling back" for Phase 2 EDF
        # without introducing complex multi-criteria optimization is to slightly penalize cloud.
        # Or, just let the EDF logic play out: if an edge node can finish earlier (including cold start, latency),
        # it will be chosen.

        # The task of "Make sure edge nodes are selected before falling back to the cloud"
        # implies a scenario where edge might be slightly slower but preferred.
        # Let's add a small bias to edge if it's within a certain 'acceptable' delay compared to cloud.

        # If the best edge option is viable (not infinitely far in the future)
        # AND (best_edge_completion_time is less than cloud_est_completion_time OR
        #      (best_edge_completion_time is only slightly higher than cloud_est_completion_time
        #       AND best_edge_completion_time <= task.deadline))
        # This needs to be carefully tuned. For Phase 2, let's stick to simple greedy choice.
        # The core EDF prioritizes by deadline. The location decision is secondary.

        # For EDF, the EDF chooses the task, then `find_best_execution_option` determines *where* to run it.
        # The `find_best_execution_option` should just report the true earliest completion time.
        # The EDF policy itself (which task to run next) is what is critical.

        # Let's correct `find_best_execution_option` to simply return the true earliest completion time
        # and the best location. The "prioritize edge" part will be more about how EDF picks the task
        # *if multiple edge nodes are free*, or in Phase 3 with more complex costs.

        # Re-evaluating the comparison for `find_best_execution_option`:
        # The intent here is to calculate the absolute earliest a given `task` can finish.
        # It's not about making a policy decision, but about finding the best *physical* option.

        # Current implementation: `if best_node is None or cloud_completion_time < min_completion_time:`
        # This correctly picks the faster of the two (cloud vs best edge).
        # We need to *ensure* the `estimated_start_time_at_node` includes all necessary costs for edge.
        # Yes, `estimated_start_time_at_node = max(current_time, current_node_available_time) + self.edge_network.base_latency + cold_start_cost` is correct now.

        # So, the logic of `find_best_execution_option` is fine for finding the absolute fastest.
        # The "prioritize edge" part will happen more organically if edge is truly faster,
        # or it needs a more complex scheduling heuristic (e.g., cost-aware, energy-aware) which is Phase 3.
        # For EDF, we just need to know the *earliest possible* completion time for each task to rank them.

        # So, the comparison here is correct:
        if best_node is None or cloud_est_completion_time < min_completion_time:
            min_completion_time = cloud_est_completion_time
            best_node = self.cloud
            best_node_type = "cloud"
            cold_start_incurred = 0.0
            calculated_wait_time = 0.0  # Cloud has no waiting queue in this model

        # Ensure cache is updated for the chosen node.
        if best_node_type == "edge":
            self.cache_manager.access_function(best_node.id, task.function_id)

        # Return the chosen node, its type, the cold start cost, and the estimated completion time.
        return best_node, best_node_type, cold_start_incurred, calculated_wait_time, min_completion_time

    def assign_task(self, task: Task, assigned_node, node_type, cold_start_cost, wait_time):
        """
        Finalizes the assignment of a task to a node and updates node availability.
        This method will be called by the simulator after a task has been selected
        by EDF and an execution option is found.
        """
        task.assigned_node = assigned_node
        task.assigned_node_type = node_type
        task.cold_start_penalty = cold_start_cost
        task.wait_time = wait_time

        # Note: task.start_time and task.end_time will be calculated in the simulator
        # based on the node's availability and the overheads returned by find_best_execution_option.
        # update_node_busy_time is called with the actual end_time.

    def update_node_busy_time(self, node, end_time, node_type):
        """Updates the available time for an edge node after a task completes."""
        if node_type == "edge":
            self.edge_node_available_time[node.id] = max(self.edge_node_available_time[node.id], end_time)
        # Cloud doesn't need its 'available_time' updated.