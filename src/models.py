# src/models.py

import collections
import networkx as nx  # Used for DAG analysis in Workflow


class Function:
    def __init__(self, id, name, runtime=1.0):
        self.id = id
        self.name = name
        self.runtime = runtime  # Base execution time of the function


class Task:
    def __init__(self, id, name, function_id, runtime, deadline, dependencies):
        self.id = id
        self.name = name
        self.function_id = function_id
        self.runtime = runtime  # Specific runtime for this task (can be from function or overridden)
        self.deadline = deadline  # Absolute time by which the task must complete
        self.dependencies = set(dependencies)  # Set of task IDs that must complete before this one starts

        self.ready_time = None  # Time when all dependencies are met and task is ready for scheduling
        self.start_time = None  # Actual simulation time when execution begins
        self.end_time = None  # Actual simulation time when execution completes
        self.assigned_node = None  # Reference to EdgeNode or Cloud object
        self.assigned_node_type = None  # "edge" or "cloud"
        self.cold_start_penalty = 0  # Penalty incurred for this specific task
        self.wait_time = 0  # Time spent waiting in queue after being ready

        # For Critical Path Analysis (Phase 2 - Initialized for consistency)
        self.est = 0
        self.eft = 0
        self.lst = float('inf')
        self.lft = float('inf')
        self.slack = float('inf')
        self.on_critical_path = False


class Workflow:
    def __init__(self, id, name, tasks, deadline):
        self.id = id
        self.name = name
        self.tasks = {task.id: task for task in tasks}  # Dictionary for easy lookup
        self.submission_time = None  # Time when the workflow is submitted to the system
        self.completion_time = None  # Time when all tasks in the workflow are completed
        self.deadline = deadline  # Absolute deadline for the entire workflow

        self.critical_path_tasks = set()  # To store task IDs on critical path (for Phase 2)
        # We'll compute the critical path later in parser or constructor after all tasks are added
        self.dependency_graph = None  # networkx.DiGraph for workflow dependencies

    def _compute_dependencies_graph(self):
        """Builds a NetworkX directed graph for task dependencies."""
        self.dependency_graph = nx.DiGraph()
        for task_id, task in self.tasks.items():
            self.dependency_graph.add_node(task_id, task_obj=task)
            for dep_id in task.dependencies:
                self.dependency_graph.add_edge(dep_id, task_id)

    def _compute_critical_path(self):
        """Calculates EST, EFT, LST, LFT, and slack for all tasks, and identifies critical path."""
        if not self.dependency_graph:
            self._compute_dependencies_graph()

        # Calculate EST and EFT (Forward Pass)
        # Ensure all tasks are processed in topological order
        try:
            for node_id in nx.topological_sort(self.dependency_graph):
                task = self.tasks[node_id]
                est = 0
                for pred_id in self.dependency_graph.predecessors(node_id):
                    pred_task = self.tasks[pred_id]
                    est = max(est, pred_task.eft)  # EST is max of EFT of predecessors
                task.est = est
                task.eft = task.est + task.runtime
        except nx.NetworkXUnfeasible:
            print(f"Warning: Workflow {self.name} (ID: {self.id}) has a cycle. Critical path calculation skipped.")
            return

        # Calculate LST and LFT (Backward Pass)
        # Determine the maximum finish time for the workflow
        workflow_end_time = 0
        for task in self.tasks.values():
            if task.eft > workflow_end_time:
                workflow_end_time = task.eft

        # If a workflow deadline is provided, the effective end time for LFT calculation is its minimum
        effective_workflow_end_time = min(workflow_end_time, self.deadline) if self.deadline else workflow_end_time

        # Iterate in reverse topological order
        for node_id in reversed(list(nx.topological_sort(self.dependency_graph))):
            task = self.tasks[node_id]

            if not list(self.dependency_graph.successors(node_id)):  # If it's an end task
                task.lft = effective_workflow_end_time
            else:
                lft = float('inf')
                for succ_id in self.dependency_graph.successors(node_id):
                    succ_task = self.tasks[succ_id]
                    lft = min(lft, succ_task.lst)  # LFT is min of LST of successors
                task.lft = lft

            task.lst = task.lft - task.runtime
            task.slack = task.lft - task.eft

            if abs(task.slack) < 1e-9:  # Consider floating point precision for 0 slack
                task.on_critical_path = True
                self.critical_path_tasks.add(task.id)


class EdgeNode:
    def __init__(self, id):
        self.id = id
        self.available_time = 0.0  # Time when this node becomes free to start a new task

    def get_available_time(self):
        return self.available_time

    def set_available_time(self, time):
        self.available_time = time


class Cloud:
    def __init__(self, latency):
        self.id = "cloud"
        self.latency = latency  # Additional latency for cloud access (network)
        # Cloud is assumed to have infinite capacity and always available with zero cold start.
        # No 'available_time' tracking needed for scheduling purposes.


class WorkflowStats:
    """Class to collect and store statistics for a single workflow."""

    def __init__(self, workflow_id, workflow_name):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.total_tasks = 0
        self.tasks_completed_on_time = 0
        self.tasks_executed_on_edge = 0
        self.tasks_executed_on_cloud = 0
        self.total_edge_wait_time = 0
        self.edge_tasks_processed_for_wait_time = 0  # Counter for tasks where wait time was recorded
        self.cold_starts = 0
        self.workflow_completed_within_deadline = False  # Tracks if *all* tasks finished by workflow deadline

    def add_task_execution_detail(self, assigned_node_type, completed_on_time, wait_time=0):
        if assigned_node_type == 'edge':
            self.tasks_executed_on_edge += 1
            self.total_edge_wait_time += wait_time
            self.edge_tasks_processed_for_wait_time += 1
        elif assigned_node_type == 'cloud':
            self.tasks_executed_on_cloud += 1

        if completed_on_time:
            self.tasks_completed_on_time += 1

    def increment_cold_starts(self):
        self.cold_starts += 1

    def calculate_average_edge_wait_time(self):
        return self.total_edge_wait_time / self.edge_tasks_processed_for_wait_time if self.edge_tasks_processed_for_wait_time > 0 else 0

    def print_stats(self):
        print(f"\n--- Workflow Statistics: {self.workflow_name} (ID: {self.workflow_id}) ---")
        print(f"Total tasks in workflow: {self.total_tasks}")
        print(f"Tasks completed on time: {self.tasks_completed_on_time} / {self.total_tasks}")
        print(f"Tasks executed on Edge: {self.tasks_executed_on_edge}")
        print(f"Tasks executed on Cloud: {self.tasks_executed_on_cloud}")
        print(f"Average wait time in Edge queues: {self.calculate_average_edge_wait_time():.4f}")
        print(f"Total Cold Starts for Workflow: {self.cold_starts}")
        print(f"Workflow Completed within Deadline: {'Yes' if self.workflow_completed_within_deadline else 'No'}")
        print("--------------------------------------------------")