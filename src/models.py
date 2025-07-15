import collections
import networkx as nx
from typing import Dict, Set, Optional, List


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

    def __init__(self, id, name, function_id, runtime, deadline, dependencies: Set[str], workflow_id: str):
        self.id = id
        self.name = name
        self.function_id = function_id
        self.runtime = runtime
        self.deadline = deadline
        self.dependencies = set(dependencies)


        self.ready_time = None
        self.start_time = None
        self.end_time = None
        self.assigned_node: Optional['EdgeNode' | 'Cloud'] = None
        self.assigned_node_type: Optional[str] = None
        self.cold_start_penalty = 0.0
        self.wait_time = 0.0
        self.on_critical_path = False
        self.workflow_id = workflow_id
        self.workflow_instance: Optional['Workflow'] = None


class EdgeNode:
    """
    Represents an edge computing node with limited capacity and a function cache.
    """

    def __init__(self, id, capacity: int = 1, cache_size: int = 5):
        self.id = id
        self.capacity = capacity
        self.occupied_slots_finish_times: List[float] = []

        self.function_cache: collections.OrderedDict[str, bool] = collections.OrderedDict()
        self.cache_size = cache_size

    def get_earliest_available_slot_time(self, current_simulation_time: float) -> float:
        """
        Returns the earliest time a slot becomes available on this node.
        This also implicitly cleans up `occupied_slots_finish_times`.
        """
        self.occupied_slots_finish_times = [
            t_end for t_end in self.occupied_slots_finish_times if t_end > current_simulation_time
        ]

        if len(self.occupied_slots_finish_times) < self.capacity:
            return current_simulation_time
        else:
            return min(self.occupied_slots_finish_times)

    def get_current_load(self, current_simulation_time: float) -> int:
        """Returns the number of functions currently being executed (occupying slots)."""
        self.occupied_slots_finish_times = [
            t_end for t_end in self.occupied_slots_finish_times if t_end > current_simulation_time
        ]
        return len(self.occupied_slots_finish_times)

    def is_busy(self, current_time: float) -> bool:
        """Checks if the node is busy at the current_time (i.e., at max capacity)."""
        return self.get_current_load(current_time) >= self.capacity

    def assign_task_to_slot(self, function_id: str, task_end_time: float):
        """
        Assigns a task to an available slot on this edge node.
        This method is called by the scheduler AFTER it has determined start/end times.
        It manages the node's internal state regarding occupied slots and function cache.
        """
        self.occupied_slots_finish_times.append(task_end_time)
        self.occupied_slots_finish_times.sort()

        if function_id in self.function_cache:
            self.function_cache.move_to_end(function_id)
        else:
            if len(self.function_cache) >= self.cache_size:
                self.function_cache.popitem(last=False)
            self.function_cache[function_id] = True

    def is_cached(self, function_id: str) -> bool:
        """Checks if a function is currently in the node's local cache."""
        return function_id in self.function_cache


class Cloud:
    """
    Represents the central cloud. For simplicity, assume effectively infinite capacity
    and a constant base latency for data transfer to/from it.
    """

    def __init__(self, base_latency: float):
        self.id = "cloud"
        self.base_latency = base_latency
        pass

    def get_available_time(self, current_simulation_time: float) -> float:
        """
        Returns the earliest time the cloud can accept a new task request.
        For an infinitely parallel cloud, this is always the current simulation time.
        """
        return current_simulation_time

    def assign_task_to_slot(self, function_id: str, task_end_time: float):
        """
        Placeholder method for Cloud, as it has infinite capacity.
        The task is simply accepted.
        """
        pass


class Workflow:
    """
    Represents a directed acyclic graph (DAG) of tasks.
    """

    def __init__(self, id: str, name: str, tasks: Dict[str, Task], initial_deadline: float,
                 source_filepath: str = "", workflow_submission_time: float = 0.0):
        self.id = id
        self.name = name
        self.tasks = tasks
        self.total_tasks = len(self.tasks)
        self.initial_deadline = initial_deadline
        self.source_filepath = source_filepath

        self.completed_tasks: Set[str] = set()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.workflow_submission_time = workflow_submission_time

        # Build dependency graph
        self.dependency_graph = nx.DiGraph()
        self._compute_dependencies_graph()
        self.critical_path_tasks: Set[str] = set()
        self._compute_critical_path()

        # Link tasks back to this workflow instance
        for task in self.tasks.values():
            task.workflow_instance = self
            task.workflow_id = self.id

    def _compute_dependencies_graph(self):
        """Builds the NetworkX DAG based on task dependencies."""
        self.dependency_graph.clear()
        for task_id, task in self.tasks.items():
            self.dependency_graph.add_node(task_id, task_obj=task)
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self.dependency_graph.add_edge(dep_id, task_id)

    def _compute_critical_path(self):
        """
        Computes the critical path based on task runtimes and marks tasks on it.
        This is typically done on the template.
        """
        if not self.dependency_graph.nodes:
            return

        est = {node: 0.0 for node in self.dependency_graph.nodes}

        try:
            topo_order = list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXNoCycle:
            print(f"Warning: Cycle detected in workflow {self.id}. Critical path might be inaccurate.")
            return

        for u_id in topo_order:
            u_task = self.tasks[u_id]
            for v_id in self.dependency_graph.successors(u_id):
                v_task = self.tasks[v_id]
                est[v_id] = max(est[v_id], est[u_id] + u_task.runtime)

        makespan = 0.0
        for node_id, task in self.tasks.items():
            makespan = max(makespan, est[node_id] + task.runtime)

        lft = {node: makespan for node in self.dependency_graph.nodes}

        for u_id in reversed(topo_order):
            u_task = self.tasks[u_id]
            for v_id in self.dependency_graph.successors(u_id):
                v_task = self.tasks[v_id]
                lft[u_id] = min(lft[u_id], lft[v_id] - v_task.runtime)

        self.critical_path_tasks.clear()
        for node_id, task in self.tasks.items():
            slack = lft[node_id] - est[node_id] - task.runtime
            if abs(slack) < 1e-9:
                task.on_critical_path = True
                self.critical_path_tasks.add(task.id)
            else:
                task.on_critical_path = False

    def get_ready_tasks(self) -> List[Task]:
        """
        Returns a list of tasks that are currently ready for scheduling (all dependencies met and not yet started).
        """
        ready_tasks = []
        for task_id, task in self.tasks.items():
            dependencies_met = all(dep_id in self.completed_tasks for dep_id in task.dependencies)

            if dependencies_met and task.start_time is None:
                ready_tasks.append(task)
        return ready_tasks

    def mark_task_completed(self, task_id: str):
        """Marks a task as completed within this workflow instance."""
        self.completed_tasks.add(task_id)

    def is_completed(self) -> bool:
        """Checks if all tasks in the workflow have been completed."""
        return len(self.completed_tasks) == self.total_tasks

    def check_if_completed_within_deadline(self, completion_time: float) -> bool:
        """Checks if the entire workflow completed by its overall deadline."""
        actual_deadline = self.initial_deadline + self.workflow_submission_time
        return completion_time <= actual_deadline

    def get_successors(self, task_id: str) -> List[Task]:
        """Returns a list of successor Task objects for a given task ID."""
        successors = []
        if task_id in self.dependency_graph:
            for successor_id in self.dependency_graph.successors(task_id):
                successors.append(self.tasks[successor_id])
        return successors

    def create_instance(self, new_id: str, current_simulation_time: float):
        """
        Creates a new instance of this workflow template, resetting all execution-specific attributes.
        """
        new_tasks = {}
        for task_id, task in self.tasks.items():
            new_task = Task(
                id=task.id,
                name=task.name,
                function_id=task.function_id,
                runtime=task.runtime,
                deadline=task.deadline,
                dependencies=set(task.dependencies),
                workflow_id=new_id
            )
            new_task.on_critical_path = task.on_critical_path
            new_tasks[task_id] = new_task

        new_workflow_instance = Workflow(
            id=new_id,
            name=self.name,
            tasks=new_tasks,
            initial_deadline=self.initial_deadline,
            source_filepath=self.source_filepath,
            workflow_submission_time=current_simulation_time
        )

        for task in new_workflow_instance.tasks.values():
            task.workflow_instance = new_workflow_instance
            task.workflow_id = new_workflow_instance.id

        return new_workflow_instance


class WorkflowStats:
    """
    Stores statistics for a single workflow instance.
    """

    def __init__(self, workflow_id: str, workflow_name: str, total_tasks: int):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.total_tasks = total_tasks
        self.tasks_executed_on_edge = 0
        self.tasks_executed_on_cloud = 0
        self.cold_starts = 0
        self.total_wait_time = 0.0
        self.tasks_completed_within_deadline = 0
        self.workflow_completed_within_deadline = False
        self.task_execution_details = []

    def add_task_execution_detail(self, node_type: str, completed_on_time: bool, wait_time: float,
                                  cold_start_incurred: float):
        if node_type == "edge":
            self.tasks_executed_on_edge += 1
        elif node_type == "cloud":
            self.tasks_executed_on_cloud += 1

        if completed_on_time:
            self.tasks_completed_within_deadline += 1

        self.total_wait_time += wait_time
        if cold_start_incurred > 0:
            self.cold_starts += 1

        self.task_execution_details.append({
            "node_type": node_type,
            "completed_on_time": completed_on_time,
            "wait_time": wait_time,
            "cold_start_incurred": cold_start_incurred
        })

    def get_average_wait_time(self) -> float:
        if self.tasks_executed_on_edge + self.tasks_executed_on_cloud == 0:
            return 0.0
        return self.total_wait_time / (self.tasks_executed_on_edge + self.tasks_executed_on_cloud)

