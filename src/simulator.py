# src/simulator.py

import heapq
import time
import random
import os
from .results import plotting

from .config import (
    NUM_EDGE_NODES, EDGE_CACHE_SIZE_PER_NODE, COLD_START_PENALTY,
    EDGE_TO_EDGE_LATENCY, CLOUD_TO_EDGE_LATENCY, PEGASUS_WORKFLOW_FILEPATHS,
    ADJACENCY_MATRIX, SCHEDULING_POLICY, CACHE_SHARING_POLICY,
    SIMULATION_DURATION, RANDOM_SEED, PUBLIC_CACHE_FRACTION,
    PREDICTION_INTERVAL,
    # --- NEW IMPORTS FOR SCALING ---
    MIN_WORKFLOW_SUBMISSION_INTERVAL, MAX_WORKFLOW_SUBMISSION_INTERVAL,
    TOTAL_WORKFLOW_SUBMISSIONS, WORKFLOW_SELECTION_PROBABILITY
    # --- END NEW IMPORTS ---
)
from .models import Task, Workflow, EdgeNode, Cloud, WorkflowStats
from .parser import PegasusWorkflowParser
from .cache_manager import CacheManager
from .edge_network import EdgeNetwork
from .scheduler import Scheduler


class Simulator:
    def __init__(self, num_edge_nodes, cache_size, cold_start_penalty,
                 edge_latency, cloud_latency, workflow_filepaths,
                 adjacency_matrix, scheduling_policy, cache_sharing_policy,
                 public_cache_fraction=0.0):

        random.seed(RANDOM_SEED)

        self.current_time = 0.0
        self.event_queue = []  # Min-heap: (time, event_type, data)

        self.edge_network = EdgeNetwork(num_edge_nodes, edge_latency, adjacency_matrix)
        self.cloud = Cloud(cloud_latency)
        self.cache_manager = CacheManager(cache_size, num_edge_nodes, "LRU", cache_sharing_policy,
                                          public_cache_fraction)
        self.scheduler = Scheduler(self.edge_network, self.cloud, self.cache_manager, COLD_START_PENALTY,
                                   scheduling_policy)

        self.workflow_parser = PegasusWorkflowParser()

        # Reset workflow_parser's counter for each simulation run to ensure unique workflow IDs per run
        self.workflow_parser.workflow_counter = 0

        # Pre-parse all available workflow definitions once
        self.available_workflow_definitions = self._pre_parse_workflow_definitions(workflow_filepaths)

        self.workflows = []  # This will store actual submitted workflow instances
        self.active_workflows = {}
        self.completed_tasks_count = {}
        self.total_tasks_in_workflow = {}

        self.workflow_stats = {}  # Stores WorkflowStats objects for all completed workflows
        self.global_tasks_on_edge = 0
        self.global_tasks_on_cloud = 0
        self.global_cold_starts = 0
        self.global_workflows_completed_on_time = 0
        self.global_total_workflows_completed = 0
        self.global_edge_utilization = {i: 0.0 for i in range(num_edge_nodes)}

        # Initializing submission events for the dynamic scaling
        self._initialize_workflow_submissions()
        self._initialize_scheduling_events()  # For periodic prediction

    def _pre_parse_workflow_definitions(self, filepaths):
        """Parses all workflow XMLs once and stores them as templates."""
        definitions = []
        for filepath in filepaths:
            # Parse but don't assign unique IDs or schedule yet
            # Pass a dummy ID for initial parsing, actual unique IDs will be assigned on submission
            workflow_template = self.workflow_parser.parse_workflow_template(filepath)
            if workflow_template:
                definitions.append(workflow_template)
        if not definitions:
            raise ValueError("No workflow definitions loaded. Check PEGASUS_WORKFLOW_FILEPATHS and XMLs.")
        return definitions

    def _initialize_workflow_submissions(self):
        """
        Schedules a total number of workflow submissions dynamically.
        """
        current_submission_time = 0.0
        for i in range(TOTAL_WORKFLOW_SUBMISSIONS):
            # Choose a workflow definition based on probability or uniformly
            if WORKFLOW_SELECTION_PROBABILITY:
                filepath = random.choices(
                    list(WORKFLOW_SELECTION_PROBABILITY.keys()),
                    weights=list(WORKFLOW_SELECTION_PROBABILITY.values()),
                    k=1
                )[0]
                workflow_template = next(
                    wt for wt in self.available_workflow_definitions if wt.source_filepath == filepath)
            else:
                # Uniformly pick from available templates
                workflow_template = random.choice(self.available_workflow_definitions)

            # Assign a unique ID for this specific submission instance
            new_workflow_instance_id = f"{workflow_template.name.split('_')[0]}_{workflow_template.id.split('_')[1]}_{self.workflow_parser.workflow_counter}"
            # Create a new workflow object for this instance
            # We need a deep copy or re-parsing to ensure task states are fresh for each instance
            # For simplicity, let's re-parse for each submission or create a 'clone' method in Workflow
            # Re-parsing is safer but slower if many workflows are used.
            # Let's clone (requires a clone method in Workflow and Task).
            # If not cloning, each workflow instance needs its own fresh Task objects.

            # --- IMPORTANT: Re-parsing per instance for simplicity ---
            # This is safer than deepcopy for nested objects and their states.
            # If performance becomes an issue with many submissions, optimize this.

            # Since parse_workflow in parser.py assigns IDs dynamically,
            # we just call it. But it needs the path.
            # To avoid re-parsing the file: let's change workflow_parser to clone tasks.
            # This will require changes in parser.py and models.py (Task.clone, Workflow.clone).

            # For immediate fix and Phase 3 Goal 4, let's simplify and make the simulator
            # just pick a template and then clone the tasks from it.
            # This requires a clone method in Workflow and Task.

            # Alternative for simplicity right now: Just re-parse if we don't have clone.
            # Re-using workflow_parser for parsing instance
            # New approach: the _pre_parse_workflow_definitions returns templates with generic IDs.
            # When a new instance is created, we use a *fresh* parser call to get fresh tasks with unique IDs.
            # This ensures unique Task/Workflow IDs.
            workflow_instance = self.workflow_parser.parse_workflow(workflow_template.source_filepath)

            if workflow_instance:
                # Override the ID generated by parser with a truly unique one based on the submission count
                workflow_instance.id = f"wf_inst_{self.workflow_parser.workflow_counter}"
                # The workflow_parser.workflow_counter is already incremented in parse_workflow
                # This ensures workflow_instance.id is unique and links to workflow_parser's counter.

                # Each task also needs a unique ID based on its parent workflow instance
                for task in workflow_instance.tasks.values():
                    task.workflow_id = workflow_instance.id  # Ensure tasks link to the unique workflow instance ID

                self.workflows.append(workflow_instance)  # Add to the list of all submitted instances

                current_submission_time += random.uniform(MIN_WORKFLOW_SUBMISSION_INTERVAL,
                                                          MAX_WORKFLOW_SUBMISSION_INTERVAL)
                heapq.heappush(self.event_queue, (current_submission_time, "workflow_submit", workflow_instance.id))
                workflow_instance.submission_time = current_submission_time
                # print(f"Scheduled Workflow '{workflow_instance.name}' (ID: {workflow_instance.id}) for submission at time {current_submission_time:.2f}")

    def _initialize_scheduling_events(self):
        """
        Schedules initial periodic prediction events.
        """
        heapq.heappush(self.event_queue, (self.current_time + PREDICTION_INTERVAL, "prediction_event", None))

    def run_simulation(self):
        # ... (unchanged run_simulation logic) ...
        # (Commented out prints for cleaner overall output)
        sim_start_real_time = time.time()

        while self.event_queue and self.current_time <= SIMULATION_DURATION:
            event_time, event_type, data = heapq.heappop(self.event_queue)
            if event_time > self.current_time:
                # print(f"Time advanced from {self.current_time:.2f} to {event_time:.2f}")
                pass  # This indicates time is progressing
            self.current_time = max(self.current_time, event_time)
            # print(f"Processing event: {event_type} at {self.current_time:.2f}") # Also useful for deep debug


            if event_type == "workflow_submit":
                workflow_id = data
                workflow = next((wf for wf in self.workflows if wf.id == workflow_id), None)
                if not workflow: continue

                # print(f"Time {self.current_time:.2f}: Workflow '{workflow.name}' (ID: {workflow.id}) submitted.")
                self.active_workflows[workflow.id] = workflow
                self.workflow_stats[workflow.id] = WorkflowStats(workflow.id, workflow.name)
                self.workflow_stats[workflow.id].total_tasks = len(workflow.tasks)
                self.completed_tasks_count[workflow.id] = 0
                self.total_tasks_in_workflow[workflow.id] = len(workflow.tasks)

                for task in workflow.tasks.values():
                    if not task.dependencies:
                        task.ready_time = self.current_time
                        self.scheduler.add_ready_task(task)

                heapq.heappush(self.event_queue, (self.current_time, "try_schedule", None))

            elif event_type == "task_ready":
                task_id, workflow_id = data
                workflow = self.active_workflows.get(workflow_id)
                if not workflow: continue
                task = workflow.tasks.get(task_id)
                if not task or task.end_time is not None: continue

                task.ready_time = self.current_time
                self.scheduler.add_ready_task(task)

                heapq.heappush(self.event_queue, (self.current_time, "try_schedule", None))

            elif event_type == "task_completed":
                task_id, workflow_id = data
                workflow = self.active_workflows.get(workflow_id)
                if not workflow: continue
                task = workflow.tasks.get(task_id)
                if not task: continue

                task_completed_on_time = (task.end_time <= task.deadline)
                self.workflow_stats[workflow_id].add_task_execution_detail(
                    task.assigned_node_type, task_completed_on_time, task.wait_time
                )

                if task.cold_start_penalty > 0:
                    self.workflow_stats[workflow_id].increment_cold_starts()

                self.completed_tasks_count[workflow_id] += 1

                for successor_task in workflow.tasks.values():
                    if task.id in successor_task.dependencies:
                        successor_task.dependencies.remove(task.id)
                        if not successor_task.dependencies and successor_task.end_time is None:
                            heapq.heappush(self.event_queue,
                                           (self.current_time, "task_ready", (successor_task.id, workflow.id)))

                if self.completed_tasks_count[workflow_id] == self.total_tasks_in_workflow[workflow_id]:
                    workflow.completion_time = self.current_time

                    workflow_on_time = all(t.end_time <= t.deadline for t in workflow.tasks.values())
                    if workflow_on_time:
                        self.workflow_stats[workflow_id].workflow_completed_within_deadline = True
                    else:
                        self.workflow_stats[workflow_id].workflow_completed_within_deadline = False

                    del self.active_workflows[workflow.id]
                    del self.completed_tasks_count[workflow.id]
                    del self.total_tasks_in_workflow[workflow.id]

                heapq.heappush(self.event_queue, (self.current_time, "try_schedule", None))

            elif event_type == "try_schedule":
                while self.scheduler.ready_task_queue:
                    if self.scheduler.scheduling_policy == "EDF":
                        _deadline, _task_id_from_heap, current_task_to_schedule = self.scheduler.ready_task_queue[0]
                    elif self.scheduler.scheduling_policy == "CriticalPathFirst":
                        _priority_flag, _deadline, _task_id_from_heap, current_task_to_schedule = \
                        self.scheduler.ready_task_queue[0]
                    else:
                        _deadline, _task_id_from_heap, current_task_to_schedule = self.scheduler.ready_task_queue[0]

                    if current_task_to_schedule.workflow_id not in self.active_workflows or \
                            current_task_to_schedule.end_time is not None:
                        heapq.heappop(self.scheduler.ready_task_queue)
                        continue

                    chosen_node, chosen_node_type, cold_start_cost, wait_time, estimated_completion_time = \
                        self.scheduler.find_best_execution_option(current_task_to_schedule, self.current_time)

                    task_start_on_resource_time = max(self.current_time, \
                                                      chosen_node.get_available_time() if chosen_node_type == "edge" else self.current_time)

                    if task_start_on_resource_time > self.current_time:
                        if self.scheduler.scheduling_policy == "EDF":
                            heapq.heappush(self.scheduler.ready_task_queue,
                                           (_deadline, _task_id_from_heap, current_task_to_schedule))
                        elif self.scheduler.scheduling_policy == "CriticalPathFirst":
                            heapq.heappush(self.scheduler.ready_task_queue,
                                           (_priority_flag, _deadline, _task_id_from_heap, current_task_to_schedule))
                        else:
                            heapq.heappush(self.scheduler.ready_task_queue,
                                           (_deadline, _task_id_from_heap, current_task_to_schedule))

                        heapq.heappush(self.event_queue, (task_start_on_resource_time, "try_schedule", None))
                        break

                    heapq.heappop(self.scheduler.ready_task_queue)

                    actual_task_execution_start_time = task_start_on_resource_time
                    if chosen_node_type == "edge":
                        actual_task_execution_start_time += self.edge_network.base_latency + cold_start_cost
                    elif chosen_node_type == "cloud":
                        actual_task_execution_start_time += self.cloud.latency

                    current_task_to_schedule.start_time = actual_task_execution_start_time
                    current_task_to_schedule.end_time = current_task_to_schedule.start_time + current_task_to_schedule.runtime

                    self.scheduler.assign_task(current_task_to_schedule, chosen_node, chosen_node_type, cold_start_cost,
                                               wait_time)
                    self.scheduler.update_node_busy_time(chosen_node, current_task_to_schedule.end_time,
                                                         chosen_node_type)

                    heapq.heappush(self.event_queue, (current_task_to_schedule.end_time, "task_completed",
                                                      (current_task_to_schedule.id,
                                                       current_task_to_schedule.workflow_id)))

                    if chosen_node_type == "edge":
                        effective_busy_duration = current_task_to_schedule.runtime + cold_start_cost + self.edge_network.base_latency
                        self.global_edge_utilization[chosen_node.id] += effective_busy_duration

            elif event_type == "prediction_event":
                if self.active_workflows:
                    # Pass the active workflows dictionary to the scheduler
                    self.scheduler.trigger_predictive_cache_load(self.active_workflows, self.current_time)

                # Schedule the next prediction event if simulation is still within duration
                if self.current_time + PREDICTION_INTERVAL <= SIMULATION_DURATION:
                    heapq.heappush(self.event_queue,
                                   (self.current_time + PREDICTION_INTERVAL, "prediction_event", None))

        self.global_total_workflows_completed = len(self.workflow_stats)
        self.global_workflows_completed_on_time = sum(
            1 for ws in self.workflow_stats.values() if ws.workflow_completed_within_deadline)
        self.global_tasks_on_edge = sum(ws.tasks_executed_on_edge for ws in self.workflow_stats.values())
        self.global_tasks_on_cloud = sum(ws.tasks_executed_on_cloud for ws in self.workflow_stats.values())
        self.global_cold_starts = sum(ws.cold_starts for ws in self.workflow_stats.values())

        sim_end_real_time = time.time()

    def print_global_stats(self):
        """Prints overall simulation statistics for a single run."""
        print(f"\n--- Global Simulation Statistics ---")
        print(f"Total Workflows Completed: {self.global_total_workflows_completed}")
        print(f"Workflows Completed On Time: {self.global_workflows_completed_on_time}")
        print(
            f"Percentage of Workflows Completed On Time: {(self.global_workflows_completed_on_time / self.global_total_workflows_completed * 100):.2f}%" if self.global_total_workflows_completed > 0 else "N/A")
        print(f"Total Tasks Executed on Edge: {self.global_tasks_on_edge}")
        print(f"Total Tasks Executed on Cloud: {self.global_tasks_on_cloud}")
        print(f"Total Cold Starts (Overall): {self.global_cold_starts}")
        print("------------------------------------")
        print("Edge Node Utilization (Total Busy Time):")
        for node_id, utilization in self.global_edge_utilization.items():
            util_percent = (utilization / self.current_time) * 100 if self.current_time > 0 else 0.0
            print(f"  Edge Node {node_id}: {utilization:.2f} time units ({util_percent:.2f}%)")
        print("------------------------------------")


if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')

    policies_to_test = ["full_private", "full_public", "partial_public_private"]

    comparison_results = []

    print(f"--- Starting Comparative Simulation Runs ---")
    print(f"Scheduling Policy: {SCHEDULING_POLICY}")
    print(f"Number of Edge Nodes: {NUM_EDGE_NODES}")
    print(f"Edge Cache Size per Node: {EDGE_CACHE_SIZE_PER_NODE}")
    print(f"Workflows to process: {TOTAL_WORKFLOW_SUBMISSIONS} instances across available types.")
    print("-" * 40)

    for policy_name in policies_to_test:
        print(f"\n=== Running Simulation for Cache Sharing Policy: {policy_name} ===")

        current_simulator = Simulator(
            num_edge_nodes=NUM_EDGE_NODES,
            cache_size=EDGE_CACHE_SIZE_PER_NODE,
            cold_start_penalty=COLD_START_PENALTY,
            edge_latency=EDGE_TO_EDGE_LATENCY,
            cloud_latency=CLOUD_TO_EDGE_LATENCY,
            workflow_filepaths=PEGASUS_WORKFLOW_FILEPATHS,
            adjacency_matrix=ADJACENCY_MATRIX,
            scheduling_policy=SCHEDULING_POLICY,
            cache_sharing_policy=policy_name,
            public_cache_fraction=PUBLIC_CACHE_FRACTION
        )
        current_simulator.run_simulation()

        results_for_policy = {
            "policy_name": policy_name,
            "total_workflows_completed": current_simulator.global_total_workflows_completed,
            "workflows_completed_on_time": current_simulator.global_workflows_completed_on_time,
            "tasks_on_edge": current_simulator.global_tasks_on_edge,
            "tasks_on_cloud": current_simulator.global_tasks_on_cloud,
            "total_cold_starts": current_simulator.global_cold_starts,
            "total_simulation_time": current_simulator.current_time,
            "edge_utilization_data": current_simulator.global_edge_utilization
        }
        comparison_results.append(results_for_policy)
        current_simulator.print_global_stats()  # Print stats for this individual run

    print("\n--- All Comparative Simulations Finished ---")
    print("--- Generating Comparison Plots ---")

    plotting.plot_policy_comparison(comparison_results, NUM_EDGE_NODES)
    print(f"Comparison plots saved to 'logs/' directory.")