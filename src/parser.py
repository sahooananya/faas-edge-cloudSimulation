import xml.etree.ElementTree as ET
import os
import networkx as nx
from typing import Dict, Set
from .models import Task, Workflow, Function
from .config import DEFAULT_FUNCTION_RUNTIME


class PegasusWorkflowParser:
    _workflow_counter = 0  # Class-level counter for unique workflow IDs for templates

    def __init__(self):
        self.namespaces = {'dax': 'http://pegasus.isi.edu/schema/DAX'}
        self.functions_catalog: Dict[str, Function] = {}  # To store unique functions found

    def parse_workflow_template(self, xml_filepath: str) -> Workflow:
        """
        Parses a Pegasus DAX XML file to create a Workflow template object.
        """
        try:
            tree = ET.parse(xml_filepath)
            root = tree.getroot()
        except FileNotFoundError:
            print(f"Error: Workflow file not found at {xml_filepath}")
            return None
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_filepath}: {e}")
            return None

        workflow_name = root.attrib.get('name', os.path.basename(xml_filepath).replace('.dax', ''))

        # Generate a unique template ID for the workflow
        PegasusWorkflowParser._workflow_counter += 1
        template_workflow_id = f"wf_template_{PegasusWorkflowParser._workflow_counter}"

        tasks_for_template: Dict[str, Task] = {}

        # Collect job definitions (tasks) first to resolve dependencies easily
        job_elements_by_id = {job_elem.attrib['id']: job_elem for job_elem in root.findall('dax:job', self.namespaces)}

        # Iterate through job elements to create Task objects
        for job_id, job_elem in job_elements_by_id.items():
            job_name = job_elem.attrib.get('name', job_id)
            function_id = job_elem.attrib.get('name', job_id)  # Function ID is often the job name in DAX

            # Determine runtime (from file or default)
            runtime_str = job_elem.find(".//dax:uses[type='executable']", self.namespaces)
            runtime = float(runtime_str.attrib.get('runtime',
                                                   str(DEFAULT_FUNCTION_RUNTIME))) if runtime_str is not None else DEFAULT_FUNCTION_RUNTIME

            # Add/update function in catalog
            if function_id not in self.functions_catalog:
                self.functions_catalog[function_id] = Function(function_id, function_id, runtime)
            else:
                self.functions_catalog[function_id].runtime = runtime  # Update runtime if different in DAX

            # Create Task object with a placeholder deadline for the template
            task_obj = Task(
                id=job_id,
                name=job_name,
                function_id=function_id,
                runtime=runtime,
                deadline=float('inf'),  # Placeholder for template, instance will have specific deadline
                dependencies=set(),
                workflow_id=template_workflow_id  # Link to template ID initially
            )
            tasks_for_template[job_id] = task_obj

        # Parse dependencies after all tasks are created
        for child_elem in root.findall('dax:child', self.namespaces):
            child_id = child_elem.attrib['ref']
            if child_id in tasks_for_template:
                for parent_elem in child_elem.findall('dax:parent', self.namespaces):
                    parent_id = parent_elem.attrib['ref']
                    if parent_id in tasks_for_template:
                        tasks_for_template[child_id].dependencies.add(parent_id)

        # Calculate an overall initial deadline for the workflow template based on its ideal makespan
        # Temporarily create a dummy workflow instance to compute critical path makespan
        # This allows using the Workflow's own critical path calculation logic
        # Note: The ID here is temporary, as this is just for template analysis.
        temp_workflow_for_makespan = Workflow("temp_wf_id", "temp_wf_name", tasks_for_template, float('inf'))

        ideal_makespan = 0.0
        if temp_workflow_for_makespan.dependency_graph.nodes:
            # Recalculate EST/LFT to find overall makespan based on runtime only
            est_temp = {node: 0.0 for node in temp_workflow_for_makespan.dependency_graph.nodes}
            try:
                topo_order_temp = list(nx.topological_sort(temp_workflow_for_makespan.dependency_graph))
            except nx.NetworkXNoCycle:
                topo_order_temp = []

            for u_id in topo_order_temp:
                u_task = temp_workflow_for_makespan.tasks[u_id]
                for v_id in temp_workflow_for_makespan.dependency_graph.successors(u_id):
                    est_temp[v_id] = max(est_temp[v_id], est_temp[u_id] + u_task.runtime)

            for node_id, task in temp_workflow_for_makespan.tasks.items():
                ideal_makespan = max(ideal_makespan, est_temp[node_id] + task.runtime)

        # Set the initial deadline as a multiple of the ideal makespan
        # A factor like 1.5x or 2.0x is common to allow for some slack.
        # If ideal_makespan is 0 (e.g., single task workflow), use a reasonable default.
        deadline_slack_factor = 2.0  # Can be moved to config.py
        calculated_workflow_template_deadline = ideal_makespan * deadline_slack_factor if ideal_makespan > 0 else 10000.0  # Default in ms

        print(f"Parsed Workflow Template '{workflow_name}' (ID: {template_workflow_id}):")
        print(f"  Min Ideal Makespan: {ideal_makespan:.2f} ms")
        print(f"  Calculated Deadline (Template): {calculated_workflow_template_deadline:.2f} ms")

        workflow_template = Workflow(
            template_workflow_id,
            workflow_name,
            tasks_for_template,
            calculated_workflow_template_deadline,
            source_filepath=xml_filepath
        )

        return workflow_template
