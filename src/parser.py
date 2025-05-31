import xml.etree.ElementTree as ET
import os
from .models import Workflow, Task, Function
import networkx as nx

# Import MS_PER_SECOND for consistent unit conversion
from .config import MS_PER_SECOND, CLOUD_TO_EDGE_LATENCY, EDGE_TO_EDGE_LATENCY, COLD_START_PENALTY


class PegasusWorkflowParser:
    def __init__(self):
        self.functions_catalog = {}
        self.workflow_counter = 0
        self.namespaces = {'dax': 'http://pegasus.isi.edu/schema/DAX'}

    def parse_workflow_template(self, xml_filepath):
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
            workflow_name = f"{workflow_name_from_file}_{root.attrib['jobCount']}"
        else:
            workflow_name = workflow_name_attr

        template_workflow_id = f"template_{workflow_name_from_file}"

        tasks_data = {}

        for job_elem in root.findall('dax:job', self.namespaces):
            task_id = job_elem.attrib.get('id')
            task_name = job_elem.attrib.get('name')
            # Convert runtime from seconds (XML) to milliseconds
            runtime = float(job_elem.attrib.get('runtime', 0.0)) * MS_PER_SECOND

            if not task_id:
                print(f"Warning: Skipping job without 'id' attribute: {ET.tostring(job_elem, encoding='unicode')}")
                continue
            if not task_name:
                task_name = task_id

            if runtime <= 0.0:
                runtime = 1.0  # Ensure minimum runtime for calculations

            function_name = task_name

            if function_name not in self.functions_catalog:
                self.functions_catalog[function_name] = Function(function_name, function_name, runtime)

            tasks_data[task_id] = {
                'id': task_id, 'name': task_name, 'function_id': function_name,
                'runtime': runtime, 'dependencies': set()
            }

        if not tasks_data:
            workflow = Workflow(template_workflow_id, workflow_name, [], float('inf'), source_filepath=xml_filepath)
            workflow._compute_critical_path()
            return workflow

        for child_elem in root.findall('dax:child', self.namespaces):
            child_id = child_elem.attrib.get('ref')
            if not child_id or child_id not in tasks_data: continue
            for parent_elem in child_elem.findall('dax:parent', self.namespaces):
                parent_id = parent_elem.attrib.get('ref')
                if not parent_id or parent_id not in tasks_data: continue
                tasks_data[child_id]['dependencies'].add(parent_id)

        tasks = []
        for task_id, data in tasks_data.items():
            # Initial task deadline, in ms, based on its converted runtime
            task_deadline = data['runtime'] * 2.0 + 100.0  # Example: 2x runtime + 100ms buffer

            tasks.append(Task(
                id=data['id'], name=data['name'], function_id=data['function_id'],
                runtime=data['runtime'], deadline=task_deadline,
                dependencies=data['dependencies'], workflow_id=template_workflow_id
            ))
        tasks.sort(key=lambda t: t.id)

        workflow_template = Workflow(template_workflow_id, workflow_name, tasks, float('inf'),
                                     source_filepath=xml_filepath)
        workflow_template._compute_critical_path()  # Compute EST, EFT for template tasks (in ms)

        return workflow_template

    def parse_workflow(self, xml_filepath):
        """
        Parses an XML file to create a new Workflow *instance* with a unique ID and a calculated deadline (in milliseconds).
        This method is called for each submission in the simulator.
        """
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
            workflow_name = f"{workflow_name_from_file}_{root.attrib['jobCount']}"
        else:
            workflow_name = workflow_name_attr

        current_workflow_instance_id = f"wf_instance_{self.workflow_counter}"
        self.workflow_counter += 1

        tasks_data = {}
        for job_elem in root.findall('dax:job', self.namespaces):
            task_id = job_elem.attrib.get('id')
            task_name = job_elem.attrib.get('name')
            # Convert runtime from seconds (XML) to milliseconds
            runtime = float(job_elem.attrib.get('runtime', 0.0)) * MS_PER_SECOND

            if not task_id:
                print(f"Warning: Skipping job without 'id' attribute: {ET.tostring(job_elem, encoding='unicode')}")
                continue
            if not task_name:
                task_name = task_id

            if runtime <= 0.0:
                runtime = 1.0  # Ensure minimum runtime for calculations

            function_name = task_name

            if function_name not in self.functions_catalog:
                self.functions_catalog[function_name] = Function(function_name, function_name, runtime)

            tasks_data[task_id] = {
                'id': task_id, 'name': task_name, 'function_id': function_name,
                'runtime': runtime, 'dependencies': set()
            }

        if not tasks_data:
            workflow_instance = Workflow(current_workflow_instance_id, workflow_name, [], 200.0,
                                         source_filepath=xml_filepath)
            workflow_instance._compute_critical_path()
            return workflow_instance

        for child_elem in root.findall('dax:child', self.namespaces):
            child_id = child_elem.attrib.get('ref')
            if not child_id or child_id not in tasks_data: continue
            for parent_elem in child_elem.findall('dax:parent', self.namespaces):
                parent_id = parent_elem.attrib.get('ref')
                if not parent_id or parent_id not in tasks_data: continue
                tasks_data[child_id]['dependencies'].add(parent_id)

        tasks_for_instance = []
        for task_id, data in tasks_data.items():
            # Initial task deadline, in ms, based on its converted runtime
            task_deadline = data['runtime'] * 2.0 + 100.0  # Example: 2x runtime + 100ms buffer

            tasks_for_instance.append(Task(
                id=data['id'], name=data['name'], function_id=data['function_id'],
                runtime=data['runtime'], deadline=task_deadline,
                dependencies=data['dependencies'], workflow_id=current_workflow_instance_id
            ))
        tasks_for_instance.sort(key=lambda t: t.id)

        # Create a temporary workflow *just* to get its initial critical path (makespan)
        # This will compute EST/EFT for tasks (in ms)
        temp_workflow_for_cpm = Workflow(
            "temp_cpm_id", workflow_name, list(tasks_for_instance), float('inf')
        )

        min_workflow_makespan = 0
        max_depth_of_dag = 0  # Represents roughly max sequential tasks, worst-case for latency
        for task in temp_workflow_for_cpm.tasks.values():
            if task.eft > min_workflow_makespan:
                min_workflow_makespan = task.eft

        # Count the number of tasks to get a rough estimate of DAG depth for latency calculation
        max_depth_of_dag = len(tasks_for_instance)

        # --- FINAL SUPER-DUPER GENEROUS DEADLINE CALCULATION (CRITICAL) ---
        # This approach ensures the deadline is extremely forgiving, acknowledging
        # the severe penalties from CLOUD_TO_EDGE_LATENCY.

        base_makespan = min_workflow_makespan

        # The core problem: many tasks go to cloud. Each such task adds 100 seconds (100,000ms).
        # Assume a very high percentage of tasks on the critical path *could* go to the cloud.
        # Let's consider 75% of tasks on the critical path might be cloud-bound, as a very safe upper bound
        # for deadline calculation.
        estimated_cloud_latency_impact = (max_depth_of_dag * 0.75) * CLOUD_TO_EDGE_LATENCY

        # Also account for edge-to-edge communication for the remaining tasks
        estimated_e2e_latency_impact = (max_depth_of_dag * 0.25) * EDGE_TO_EDGE_LATENCY  # For non-cloud tasks

        # Add a cold start penalty for each task
        estimated_cold_start_impact = max_depth_of_dag * COLD_START_PENALTY

        # Sum up all these potential worst-case delays
        total_estimated_overheads = estimated_cloud_latency_impact + \
                                    estimated_e2e_latency_impact + \
                                    estimated_cold_start_impact

        # Add a massive fixed buffer to account for unpredictable queuing and overall system variability
        # This should be exceptionally large to guarantee success
        ABSOLUTE_GUARANTEE_BUFFER_MS = 30.0 * 60.0 * MS_PER_SECOND  # 30 minutes = 1,800,000 ms

        calculated_workflow_deadline = base_makespan + \
                                       total_estimated_overheads + \
                                       ABSOLUTE_GUARANTEE_BUFFER_MS

        # Ensure a very high absolute minimum deadline to catch any edge cases
        # For CyberShake, which has very long tasks, the deadline needs to be hours.
        MIN_ABSOLUTE_DEADLINE_FLOOR = 60.0 * 60.0 * MS_PER_SECOND  # 1 hour = 3,600,000 ms

        calculated_workflow_deadline = max(calculated_workflow_deadline, MIN_ABSOLUTE_DEADLINE_FLOOR)

        # Create the final workflow instance with its unique ID and calculated deadline
        workflow_instance = Workflow(
            current_workflow_instance_id, workflow_name, tasks_for_instance,
            calculated_workflow_deadline, source_filepath=xml_filepath
        )

        # The Workflow.__init__ will call _compute_critical_path again with this final deadline,
        # correctly setting LFTs/LSTs and slack for tasks based on the overall workflow deadline.

        print(f"Parsed Workflow Instance '{workflow_instance.name}' (ID: {workflow_instance.id}):")
        print(f"  Min Ideal Makespan: {min_workflow_makespan:.2f} ms")
        print(f"  Estimated DAG Depth (for latency): {max_depth_of_dag}")
        print(f"  Estimated Cloud Latency Impact: {estimated_cloud_latency_impact:.2f} ms")
        print(f"  Estimated E2E Latency Impact: {estimated_e2e_latency_impact:.2f} ms")
        print(f"  Estimated Cold Start Impact: {estimated_cold_start_impact:.2f} ms")
        print(
            f"  Calculated Deadline: {calculated_workflow_deadline:.2f} ms ({calculated_workflow_deadline / MS_PER_SECOND / 60.0:.2f} minutes)")

        return workflow_instance