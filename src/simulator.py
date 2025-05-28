# src/simulator.py

import heapq
import time
import os
import random

from .config import (
    NUM_EDGE_NODES, EDGE_CACHE_SIZE_PER_NODE, COLD_START_PENALTY,
    EDGE_LATENCY, CLOUD_LATENCY, PEGASUS_WORKFLOW_FILEPATHS,
    ADJACENCY_MATRIX, SCHEDULING_POLICY, CACHE_SHARING_POLICY,
    SIMULATION_DURATION, RANDOM_SEED
)
from .models import Task, Workflow, EdgeNode, Cloud, WorkflowStats
from .parser import PegasusWorkflowParser
from .cache_manager import CacheManager
from .edge_network import EdgeNetwork
from .scheduler import Scheduler


class Simulator:
    def __init__(self, num_edge_nodes, cache_size, cold_start_penalty,
                 edge_latency, cloud_latency, workflow_filepaths,
                 adjacency_matrix, scheduling_policy, cache_sharing_policy):

        random.seed(RANDOM_SEED)

        self.current_time = 0.0
        self.event_queue = []  # Min-heap: (time, event_type, data)

        self.edge_network = EdgeNetwork(num_edge_nodes, edge_latency)
        self.cloud = Cloud(cloud_latency)
        self.cache_manager = CacheManager(cache_size, num_edge_nodes, "LRU", cache_sharing_policy)
        # Pass necessary config values to scheduler for its internal calculations
        self.scheduler = Scheduler(self.edge_network, self.cloud, self.cache_manager, COLD_START_PENALTY,
                                   scheduling_policy)

        self.workflow_parser = PegasusWorkflowParser()

        self.workflows = []
        self.active_workflows = {}
        self.completed_tasks_count = {}  # {workflow_id: count}
        self.total_tasks_in_workflow = {}  # {workflow_id: total_count}

        self.workflow_stats = {}
        self.global_tasks_on_edge = 0
        self.global_tasks_on_cloud = 0
        self.global_cold_starts = 0
        self.global_workflows_completed_on_time = 0
        self.global_total_workflows_completed = 0
        self.global_edge_utilization = {i: 0.0 for i in range(num_edge_nodes)}

        self._load_workflows(workflow_filepaths)
        self._initialize_scheduling_events()

    def _load_workflows(self, filepaths):
        """Loads workflows from XML files and schedules their submission."""
        for i, filepath in enumerate(filepaths):
            workflow = self.workflow_parser.parse_workflow(filepath)
            if workflow:  # Only add if parsing was successful
                self.workflows.append(workflow)

                submission_delay = i * 50.0
                submission_time = submission_delay + random.uniform(0, 10.0)

                heapq.heappush(self.event_queue, (submission_time, "workflow_submit", workflow.id))
                workflow.submission_time = submission_time
                print(
                    f"Scheduled Workflow '{workflow.name}' (ID: {workflow.id}) for submission at time {submission_time:.2f}")

    def _initialize_scheduling_events(self):
        """
        Initializes events that trigger the scheduler.
        For EDF, we need to try scheduling whenever a resource might become free or a task becomes ready.
        A simple way is to schedule a 'try_schedule' event at each moment a node becomes free.
        """
        # For simplicity in Phase 2, we can just push a 'try_schedule' event whenever
        # a task completes, or a new workflow is submitted (which might make tasks ready).
        # The main loop itself will then query the scheduler.
        pass  # The scheduler will be called by task_ready and task_completed events.

    def run_simulation(self):
        print(f"Starting simulation. Max duration: {SIMULATION_DURATION:.2f} time units.")
        sim_start_real_time = time.time()

        while self.event_queue and self.current_time <= SIMULATION_DURATION:
            event_time, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = max(self.current_time, event_time)

            if event_type == "workflow_submit":
                workflow_id = data
                workflow = next((wf for wf in self.workflows if wf.id == workflow_id), None)
                if not workflow: continue

                print(f"Time {self.current_time:.2f}: Workflow '{workflow.name}' (ID: {workflow.id}) submitted.")
                self.active_workflows[workflow.id] = workflow
                self.workflow_stats[workflow.id] = WorkflowStats(workflow.id, workflow.name)
                self.workflow_stats[workflow.id].total_tasks = len(workflow.tasks)
                self.completed_tasks_count[workflow.id] = 0
                self.total_tasks_in_workflow[workflow.id] = len(workflow.tasks)

                # Identify initial ready tasks (no dependencies) and add them to scheduler's ready queue
                for task in workflow.tasks.values():
                    if not task.dependencies:
                        task.ready_time = self.current_time
                        self.scheduler.add_ready_task(task)

                # After submitting and adding initial tasks, trigger a scheduling attempt
                heapq.heappush(self.event_queue, (self.current_time, "try_schedule", None))


            elif event_type == "task_ready":  # This event will now simply add tasks to scheduler's queue
                task_id, workflow_id = data
                workflow = self.active_workflows.get(workflow_id)
                if not workflow: continue
                task = workflow.tasks.get(task_id)
                if not task or task.end_time is not None: continue

                task.ready_time = self.current_time  # Update ready time
                self.scheduler.add_ready_task(task)

                # Immediately try to schedule if a task becomes ready
                heapq.heappush(self.event_queue, (self.current_time, "try_schedule", None))


            elif event_type == "task_completed":
                task_id, workflow_id = data
                workflow = self.active_workflows.get(workflow_id)
                if not workflow: continue
                task = workflow.tasks.get(task_id)
                if not task: continue

                # Update workflow stats for this task
                task_completed_on_time = (task.end_time <= task.deadline)
                self.workflow_stats[workflow_id].add_task_execution_detail(
                    task.assigned_node_type, task_completed_on_time, task.wait_time
                )

                if task.cold_start_penalty > 0:
                    self.workflow_stats[workflow_id].increment_cold_starts()
                    self.global_cold_starts += 1

                if task.assigned_node_type == 'edge':
                    self.global_tasks_on_edge += 1
                else:
                    self.global_tasks_on_cloud += 1

                self.completed_tasks_count[workflow_id] += 1

                # Check for dependent tasks
                for successor_task in workflow.tasks.values():
                    if task.id in successor_task.dependencies:
                        successor_task.dependencies.remove(task.id)
                        if not successor_task.dependencies and successor_task.end_time is None:
                            heapq.heappush(self.event_queue,
                                           (self.current_time, "task_ready", (successor_task.id, workflow.id)))
                            # The task is now ready, will be added to scheduler's queue by 'task_ready' handler

                # Check if the entire workflow is complete
                if self.completed_tasks_count[workflow_id] == self.total_tasks_in_workflow[workflow_id]:
                    workflow.completion_time = self.current_time
                    self.global_total_workflows_completed += 1

                    workflow_on_time = all(t.end_time <= t.deadline for t in workflow.tasks.values())
                    if workflow_on_time:
                        self.global_workflows_completed_on_time += 1
                        self.workflow_stats[workflow_id].workflow_completed_within_deadline = True
                    else:
                        self.workflow_stats[workflow_id].workflow_completed_within_deadline = False

                    self.workflow_stats[workflow_id].print_stats()
                    del self.active_workflows[workflow.id]
                    del self.completed_tasks_count[workflow.id]
                    del self.total_tasks_in_workflow[workflow.id]

                # After a task completes, resources might be free, so try to schedule more.
                heapq.heappush(self.event_queue, (self.current_time, "try_schedule", None))


            elif event_type == "try_schedule":
                # This is where the scheduler is invoked to make decisions.
                # Find available edge nodes
                available_edge_nodes = [
                    node for node in self.edge_network.get_all_edge_nodes()
                    if self.scheduler.edge_node_available_time[node.id] <= self.current_time
                ]

                # While there are available edge nodes OR we need to send to cloud
                # And there are tasks in the ready queue
                # We prioritize assigning to edge nodes if available.

                # Iterate as long as we have tasks to schedule and resources available.
                while True:
                    # Get the highest priority task from the scheduler's ready queue (EDF)
                    # Need to peek and only pop if we can assign it
                    next_task_candidate = self.scheduler.get_next_task_to_schedule(self.current_time)

                    if not next_task_candidate:
                        break  # No more ready tasks

                    # Check if this task actually belongs to an active workflow and is not already completed
                    if next_task_candidate.workflow_id not in self.active_workflows or \
                            next_task_candidate.end_time is not None:
                        # This task is stale, remove it and try again (EDF queue might have old entries)
                        heapq.heappop(self.scheduler.ready_task_queue)  # Remove the invalid entry
                        continue  # Try to get the next valid task

                    # Find the best execution option for this task (edge or cloud)
                    chosen_node, chosen_node_type, cold_start_cost, wait_time, estimated_completion_time = \
                        self.scheduler.find_best_execution_option(next_task_candidate, self.current_time)

                    # Check if the chosen resource (edge or cloud) is available or if we need to wait
                    # If it's an edge node, is it available by now?
                    # If it's the cloud, it's always "available" instantly for dispatch.

                    if chosen_node_type == "edge":
                        node_available_time = self.scheduler.edge_node_available_time[chosen_node.id]
                        if estimated_completion_time > self.current_time and node_available_time > self.current_time:
                            # The chosen edge node is busy.
                            # We should re-evaluate if the task can be sent to another node (cloud)
                            # Or, if this task is going to this specific edge node,
                            # we need to schedule a future 'try_schedule' or similar event
                            # for when this node actually becomes free.

                            # For simple EDF, if the best option is busy, we don't assign it *now*.
                            # We can either:
                            # 1. Pop this task and see if another task can be scheduled on an available node.
                            # 2. Re-schedule a 'try_schedule' event for when this best_node becomes free.

                            # Option 1 (simpler for now): Don't schedule this task immediately if its best node is busy.
                            # It will remain in the ready_task_queue (or be re-added) and be picked up later.
                            # Break this inner loop as we can't schedule the highest priority task immediately.
                            # This implies tasks higher in the EDF queue might block tasks lower down,
                            # even if lower tasks could run on a different free node.
                            # A more advanced EDF would check multiple nodes.

                            # Let's refine this: If the highest priority task's *best* execution option
                            # means waiting on a busy resource, then we cannot assign it *right now*.
                            # We should *not* pop it from the ready_task_queue yet.
                            # Instead, we should exit this `try_schedule` loop and wait for the resource
                            # to become free (which will trigger a future 'try_schedule' event via task completion).

                            # However, if there are *other* available tasks that could run *now* on another free resource,
                            # the current EDF approach will still try to pick the highest priority *overall*.

                            # To address "Make sure edge nodes are selected before falling back to the cloud.":
                            # If an edge node is free, prioritize it.
                            # If multiple edge nodes are free, pick the best one.
                            # If no edge node is free, or all are too slow, THEN consider cloud.

                            # The `find_best_execution_option` already finds the absolute best completion time.
                            # If `chosen_node_type == "edge"` and `chosen_node.available_time > self.current_time`,
                            # it means we have to wait for that edge node. The task won't start immediately.

                            # Let's make sure we only *pop* from the ready_task_queue if we *successfully assign* a task.
                            # This means the current `next_task_candidate = self.scheduler.get_next_task_to_schedule(self.current_time)`
                            # needs to be carefully handled.

                            # Let's change the `try_schedule` loop to iterate through the *available* nodes,
                            # and for each available node, find the best task for it from the ready queue.
                            # This is a common pattern for "pull-based" scheduling.

                            # To do this, we need to sort tasks by deadline for EDF, but also consider *resource availability*.
                            # This becomes complex. A simpler EDF: when `try_schedule` fires, check if *any* resource (edge/cloud)
                            # is free to take the *highest priority READY task*.

                            # Let's simplify the `try_schedule` for EDF:
                            # While an edge node is free OR the cloud is a viable option AND there are ready tasks:

                            # Current `try_schedule` logic:
                            # 1. Get highest priority task from ready_task_queue (EDF choice).
                            # 2. Find best physical resource (edge/cloud) for that task.
                            # 3. If that resource is available NOW, assign it.
                            # 4. If that resource is busy, then the task will have to wait for that resource.
                            #    We should NOT process other, lower-priority tasks that could run *now*
                            #    if the highest priority task is waiting for a specific resource.
                            #    This is crucial for strict EDF.

                            # So, if `estimated_completion_time` (of the highest priority task on its *best* resource)
                            # implies a start time *after* `current_time` (meaning, the resource is busy for it),
                            # then we *cannot* schedule this task right now. We also cannot schedule lower priority tasks.
                            # The simulation must simply advance time to the `estimated_completion_time` of this task,
                            # or the completion time of another task that frees up a resource.

                            # Let's put a 'break' condition if the best task cannot be scheduled immediately.
                            # This will ensure strict EDF.

                            # But first, check if `next_task_candidate` is None OR if its chosen resource (edge or cloud)
                            # is busy *relative to current_time*.

                            task_start_on_resource_time = max(self.current_time,
                                                              chosen_node.get_available_time() if chosen_node_type == "edge" else self.current_time)

                            if task_start_on_resource_time > self.current_time:
                                # The best resource for the highest priority task is busy.
                                # The highest priority task cannot start right now.
                                # So, no other task can start either, because EDF prioritizes.
                                # Schedule a future 'try_schedule' event for when this resource becomes free,
                                # to re-evaluate.
                                heapq.heappush(self.event_queue, (task_start_on_resource_time, "try_schedule", None))
                                break  # Exit the try_schedule loop; nothing can run right now.

                            # If we reach here, the highest priority task CAN start now on its chosen resource.
                            # Pop it from the ready queue and assign it.
                            heapq.heappop(self.scheduler.ready_task_queue)  # Pop the assigned task

                            # Final calculations for actual task start and end times
                            # These include the latency and cold start.
                            task_start_time = task_start_on_resource_time
                            if chosen_node_type == "edge":
                                task_start_time += self.edge_network.base_latency + cold_start_cost
                            elif chosen_node_type == "cloud":
                                task_start_time += self.cloud.latency

                            task.start_time = task_start_time
                            task.end_time = task.start_time + task.runtime

                            # Assign the task within the scheduler (updates internal state like `running_tasks`)
                            self.scheduler.assign_task(next_task_candidate, chosen_node, chosen_node_type,
                                                       cold_start_cost, wait_time)

                            # Enqueue task completion event
                            heapq.heappush(self.event_queue, (task.end_time, "task_completed", (task.id, workflow.id)))

                            # Update the node's busy time for future scheduling
                            self.scheduler.update_node_busy_time(chosen_node, task.end_time, chosen_node_type)

                            # Update global edge utilization
                            if chosen_node_type == "edge":
                                effective_busy_duration = task.runtime + cold_start_cost + self.edge_network.base_latency
                                self.global_edge_utilization[chosen_node.id] += effective_busy_duration

                            # Continue the loop to see if more tasks can be scheduled now (e.g., if another resource is free)
                            # or if the current one finished quickly and can take another.
                            # This loop will break if no more tasks are ready or best resource is busy.

        sim_end_real_time = time.time()
        print(f"\nSimulation finished in {sim_end_real_time - sim_start_real_time:.2f} seconds.")
        self.print_global_stats()

        # Plotting will be for Phase 2/3
        # from .results import plotting
        # plotting.plot_workflow_completion(list(self.workflow_stats.values()))
        # plotting.plot_edge_cloud_task_ratio(list(self.workflow_stats.values()))
        # plotting.plot_edge_utilization_heatmap(self.global_edge_utilization, len(self.edge_network.edge_nodes), self.current_time)