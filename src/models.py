# src/models.py

import collections
import networkx as nx


class Function:
    """
    Represents a serverless function with its basic properties.
    """

    def __init__(self, id, name, runtime=1.0):
        self.id = id
        self.name = name
        self.runtime = runtime


class Task:
    """
    Represents a task within a workflow, corresponding to a function execution.
    """

    def __init__(self, id, name, function_id, runtime, deadline, dependencies, workflow_id):
        self.id = id
        self.name = name
        self.function_id = function_id
        self.runtime = runtime
        self.deadline = deadline  # Absolute deadline for this task
        self.dependencies = set(dependencies)  # IDs of tasks this task depends on

        # Execution related attributes
        self.ready_time = None  # Time when task becomes ready for scheduling
        self.start_time = None  # Actual start time of execution on a node
        self.end_time = None  # Actual end time of execution on a node
        self.assigned_node = None  # Reference to the EdgeNode or Cloud object
        self.assigned_node_type = None  # "edge" or "cloud"
        self.cold_start_penalty = 0  # Cold start penalty incurred if any
        self.wait_time = 0  # Time spent waiting in queue on the node

        # Workflow-related attributes
        self.workflow_id = workflow_id  # ID of the parent workflow instance

        # Critical Path Method (CPM) attributes
        self.est = 0  # Earliest Start Time
        self.eft = 0  # Earliest Finish Time
        self.lst = float('inf')  # Latest Start Time
        self.lft = float('inf')  # Latest Finish Time
        self.slack = float('inf')  # Slack time (LST - EST or LFT - EFT)
        self.on_critical_path = False  # True if task is on the critical path


class Workflow:
    """
    Represents a Directed Acyclic Graph (DAG) workflow composed of tasks.
    """

    def __init__(self, id, name, tasks, deadline, source_filepath=None):
        self.id = id
        self.name = name
        self.tasks = {task.id: task for task in tasks}  # Dictionary for easy task lookup
        self.submission_time = None  # Time when the workflow is submitted to the simulator
        self.completion_time = None  # Time when all tasks in the workflow are completed
        self.deadline = deadline  # Absolute deadline for the entire workflow
        self.source_filepath = source_filepath  # Path to the XML file this workflow was parsed from (for re-parsing instances)

        self.critical_path_tasks = set()  # Stores IDs of tasks on the critical path
        self.dependency_graph = None  # NetworkX graph representing task dependencies

        # Call critical path computation on initialization
        # This will also build the dependency_graph
        self._compute_critical_path()

    def _compute_dependencies_graph(self):
        """
        Builds a NetworkX directed graph for task dependencies.
        Edges go from predecessor to successor.
        """
        self.dependency_graph = nx.DiGraph()
        for task_id, task in self.tasks.items():
            self.dependency_graph.add_node(task_id, task_obj=task)  # Store task object as node attribute
            for dep_id in task.dependencies:
                # Add edge from dependency (parent) to current task (child)
                self.dependency_graph.add_edge(dep_id, task_id)

    def _compute_critical_path(self):
        """
        Calculates Earliest Start Time (EST), Earliest Finish Time (EFT),
        Latest Start Time (LST), Latest Finish Time (LFT), and slack for all tasks.
        Also identifies tasks that are on the critical path.
        """
        if not self.dependency_graph:
            self._compute_dependencies_graph()

        try:
            # Forward pass: Calculate EST and EFT
            for node_id in nx.topological_sort(self.dependency_graph):
                task = self.tasks[node_id]

                est = 0
                for pred_id in self.dependency_graph.predecessors(node_id):
                    pred_task = self.tasks[pred_id]
                    est = max(est, pred_task.eft)  # EST is max EFT of predecessors

                task.est = est
                task.eft = task.est + task.runtime

            # Determine the workflow's overall completion time based on EFTs
            workflow_end_time = 0
            for task in self.tasks.values():
                if task.eft > workflow_end_time:
                    workflow_end_time = task.eft

            # If the workflow has a deadline, use the minimum of its natural completion time and the deadline
            effective_workflow_end_time = min(workflow_end_time, self.deadline) if self.deadline else workflow_end_time

            # Backward pass: Calculate LFT, LST, and slack
            for node_id in reversed(list(nx.topological_sort(self.dependency_graph))):
                task = self.tasks[node_id]

                if not list(self.dependency_graph.successors(node_id)):
                    # If it's a sink node (no successors), LFT is the effective workflow end time
                    task.lft = effective_workflow_end_time
                else:
                    lft = float('inf')
                    for succ_id in self.dependency_graph.successors(node_id):
                        succ_task = self.tasks[succ_id]
                        lft = min(lft, succ_task.lst)  # LFT is min LST of successors
                    task.lft = lft

                task.lst = task.lft - task.runtime
                task.slack = task.lft - task.eft  # Slack = LFT - EFT

                # A task is on the critical path if its slack is zero (or very close to zero due to float precision)
                if abs(task.slack) < 1e-9:  # Use a small epsilon for floating point comparison
                    task.on_critical_path = True
                    self.critical_path_tasks.add(task.id)

        except nx.NetworkXUnfeasible:
            # This handles cases where the graph might have a cycle (shouldn't happen with valid DAGs)
            print(f"Warning: Workflow {self.name} (ID: {self.id}) has a cycle. Critical path calculation skipped.")
            return


class EdgeNode:
    """
    Represents an edge computing node.
    """

    def __init__(self, id, capacity=100.0, processing_rate=1.0):
        self.id = id
        self.capacity = capacity  # E.g., max concurrent functions or total processing power
        self.processing_rate = processing_rate  # How fast it processes tasks
        self.functions_in_cache = set()  # Set of function IDs currently cached
        self._available_time = 0.0  # When this node becomes free next

    def get_available_time(self):
        return self._available_time

    def set_available_time(self, time):
        self._available_time = time


class Cloud:
    """
    Represents the central cloud data center.
    """

    def __init__(self, latency):
        self.id = "cloud"
        self.latency = latency  # Latency to reach the cloud from any edge node
        self._available_time = 0.0  # Cloud is generally assumed to have infinite capacity and always available

    def get_available_time(self):
        return self._available_time  # Always ready

    def set_available_time(self, time):
        # For the cloud, this might not be strictly necessary as it's often modeled as always available.
        # However, keeping it for consistency in scheduler's update_node_busy_time.
        self._available_time = time


class WorkflowStats:
    """
    Collects and holds statistics for a single workflow instance.
    """

    def __init__(self, workflow_id, workflow_name):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.total_tasks = 0  # Will be populated after parsing
        self.tasks_executed_on_edge = 0
        self.tasks_executed_on_cloud = 0
        self.cold_starts = 0
        self.total_wait_time = 0.0
        self.tasks_completed_within_deadline = 0  # Individual task deadlines
        self.workflow_completed_within_deadline = False  # Overall workflow deadline

    def add_task_execution_detail(self, node_type: str, completed_on_time: bool, wait_time: float):
        if node_type == "edge":
            self.tasks_executed_on_edge += 1
        elif node_type == "cloud":
            self.tasks_executed_on_cloud += 1

        self.total_wait_time += wait_time

        if completed_on_time:
            self.tasks_completed_within_deadline += 1

    def increment_cold_starts(self):
        self.cold_starts += 1