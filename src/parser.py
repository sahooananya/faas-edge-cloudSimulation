# src/parser.py

import xml.etree.ElementTree as ET
import os
from .models import Workflow, Task, Function
import networkx as nx


class PegasusWorkflowParser:
    def __init__(self):
        self.functions_catalog = {}
        self.workflow_counter = 0
        # Define the namespace dictionary
        self.namespaces = {'dax': 'http://pegasus.isi.edu/schema/DAX'}  # 'dax' is just an alias

    def parse_workflow(self, xml_filepath):
        try:
            tree = ET.parse(xml_filepath)
            root = tree.getroot()
        except FileNotFoundError:
            print(f"Error: XML file not found at {xml_filepath}")
            return None
        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_filepath}: {e}")
            return None

        filename = os.path.basename(xml_filepath)
        workflow_name_from_file = os.path.splitext(filename)[0]

        workflow_name_attr = root.attrib.get('name', workflow_name_from_file)
        if workflow_name_attr == 'test' and 'jobCount' in root.attrib:
            # Use jobCount to differentiate if name is generic "test"
            workflow_name = f"{workflow_name_from_file}_{root.attrib['jobCount']}"
        else:
            workflow_name = workflow_name_attr

        workflow_id = f"wf_{self.workflow_counter}"
        self.workflow_counter += 1

        tasks_data = {}

        # --- IMPORTANT CHANGE HERE: Use the namespace prefix ---
        # Find all 'job' elements within the 'dax' namespace
        for job_elem in root.findall('dax:job', self.namespaces):
            task_id = job_elem.attrib.get('id')
            task_name = job_elem.attrib.get('name')
            runtime = float(job_elem.attrib.get('runtime', 0.0))

            if not task_id:
                print(f"Warning: Skipping job without 'id' attribute: {ET.tostring(job_elem, encoding='unicode')}")
                continue
            if not task_name:
                # print(f"Warning: Job ID {task_id} has no 'name' attribute. Using ID as name.")
                task_name = task_id

            if runtime <= 0.0:
                runtime = 1.0
                # print(f"Warning: Job ID {task_id} has invalid/zero runtime. Defaulting to {runtime}.")

            function_name = task_name

            if function_name not in self.functions_catalog:
                self.functions_catalog[function_name] = Function(function_name, function_name, runtime)

            tasks_data[task_id] = {
                'id': task_id,
                'name': task_name,
                'function_id': function_name,
                'runtime': runtime,
                'dependencies': set()
            }

        if not tasks_data:
            print(
                f"Warning: No 'job' elements found in {xml_filepath}. Workflow '{workflow_name}' (ID: {workflow_id}) will have 0 tasks.")
            workflow = Workflow(workflow_id, workflow_name, [], float('inf'))
            workflow._compute_dependencies_graph()
            workflow._compute_critical_path()
            return workflow

        # --- IMPORTANT CHANGE HERE: Use the namespace prefix for 'child' ---
        for child_elem in root.findall('dax:child', self.namespaces):
            child_id = child_elem.attrib.get('ref')
            if not child_id or child_id not in tasks_data:
                continue

            # --- IMPORTANT CHANGE HERE: Use the namespace prefix for 'parent' ---
            for parent_elem in child_elem.findall('dax:parent', self.namespaces):
                parent_id = parent_elem.attrib.get('ref')
                if not parent_id or parent_id not in tasks_data:
                    continue
                tasks_data[child_id]['dependencies'].add(parent_id)

        tasks = []
        for task_id, data in tasks_data.items():
            task_deadline = data['runtime'] * 20.0 + 50.0

            tasks.append(Task(
                id=data['id'],
                name=data['name'],
                function_id=data['function_id'],
                runtime=data['runtime'],
                deadline=task_deadline,
                dependencies=data['dependencies']
            ))

        tasks.sort(key=lambda t: t.id)

        workflow_deadline = float('inf')

        workflow = Workflow(workflow_id, workflow_name, tasks, workflow_deadline)

        workflow._compute_critical_path()

        if workflow.deadline != float('inf'):
            for task in workflow.tasks.values():
                if task.lft != float('inf'):
                    task.deadline = min(task.deadline, task.lft)

        print(f"Parsed Workflow: {workflow.name} (ID: {workflow.id}) with {len(workflow.tasks)} tasks.")
        return workflow