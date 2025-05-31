# FaaS Edge-Cloud Simulator

## Project Description
This project implements a Discrete Event Simulator for Function-as-a-Service (FaaS) deployments across an Edge-Cloud continuum. It aims to model and analyze the performance of various scheduling and caching strategies for serverless functions in a distributed environment. The simulator allows for configurable network topologies, resource characteristics, and workload patterns to evaluate different deployment scenarios.

## Key Features Implemented

### Phase 1: Core Simulation Framework
- **Discrete Event Simulation Engine:** A time-ordered event queue processes simulated events.
- **Basic Models:** `Task`, `Workflow`, `EdgeNode`, `Cloud` entities with fundamental properties.
- **Workflow Parsing:** Integration with Pegasus DAX XML files to define complex scientific workflows.
- **Simple Scheduling:** Initial greedy scheduling logic to assign tasks to available resources.

### Phase 2: Enhanced Capabilities
- **Statistics Collection:** Comprehensive tracking of workflow completion times, tasks on edge/cloud, cold starts, and resource utilization.
- **Adjacent Edge Fallback:** Scheduler checks neighboring edge nodes for availability before offloading to the cloud, enhancing local resource utilization.
- **Cache-Aware Execution:** `CacheManager` with LRU (Least Recently Used) policy simulates function caching on edge nodes, impacting cold start penalties.
- **Smarter Task Scheduling (EDF):** Earliest Deadline First (EDF) policy implemented in the scheduler to prioritize tasks based on their deadlines.
- **Visualization:** Generates insightful plots (workflow completion, edge/cloud task ratio, edge utilization) to visualize simulation results.

### Phase 3: Advanced Optimizations & Scaling
- **Smarter Task Scheduling (Critical Path First - CPF):** The scheduler now prioritizes tasks on the critical path of a workflow, aiming to reduce overall workflow completion time.
- **Cache Sharing Policies:**
    - **Full Private:** Each edge node maintains its own independent cache.
    - **Full Public:** All edge nodes share a single, global cache pool.
    - **Partial Public/Private:** A hybrid approach where nodes have a private cache portion, and a portion of capacity is pooled into a public shared cache.
    - The simulation now runs comparative analyses across these policies.
- **Predictive Cache Loading (Best Strategy):** Implements an advanced look-ahead mechanism to pre-load functions into edge caches. This strategy identifies "imminent" tasks (few dependencies remaining) and critical path tasks, attempting to load their functions to the most likely execution node before they are needed.
- **Configurable Scaling:** The simulation can now run a specified total number of workflow instances (`TOTAL_WORKFLOW_SUBMISSIONS`) with configurable inter-arrival times (`MIN_WORKFLOW_SUBMISSION_INTERVAL`, `MAX_WORKFLOW_SUBMISSION_INTERVAL`). It can also optionally simulate a mix of different workflow types.

## Project Structure
faas-edge-cloud-sim/
├── src/
│   ├── init.py
│   ├── cache_manager.py          # Manages function caching (LRU, sharing policies)
│   ├── config.py                 # Simulation parameters and network configuration
│   ├── edge_network.py           # Models edge nodes and their network topology
│   ├── models.py                 # Defines core entities (Task, Workflow, EdgeNode, Cloud, Function, WorkflowStats)
│   ├── parser.py                 # Parses Pegasus DAX XML workflow definitions
│   ├── scheduler.py              # Implements task scheduling policies (EDF, CPF)
│   ├── simulator.py              # Main simulation engine, event loop
│   └── results/
│       ├── init.py
│       └── plotting.py           # Generates comparative plots and individual run plots
├── data/                         # Directory for workflow XML files (e.g., Inspiral_30.xml)
│   ├── Inspiral_30.xml
│   ├── CyberShake_30.xml
│   └── Sipht_30.xml
├── .venv/                        # Python virtual environment (if created locally)
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project documentation (this file)


## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    cd faas-edge-cloud-sim
    ```
    (If you are not using Git, just ensure you have the `faas-edge-cloud-sim` directory with all the `src` and `data` subdirectories.)

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    * **Windows (Command Prompt):**
        ```bash
        .venv\Scripts\activate
        ```
    * **Windows (PowerShell):**
        ```bash
        .venv\Scripts\Activate.ps1
        ```
    * **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Simulation

1.  **Configure Simulation Parameters:**
    Open `src/config.py` and adjust parameters such as:
    * `NUM_EDGE_NODES`
    * `EDGE_CACHE_SIZE_PER_NODE`
    * `COLD_START_PENALTY`
    * `SCHEDULING_POLICY` (e.g., "EDF", "CriticalPathFirst")
    * `CACHE_SHARING_POLICY` (Default, will be overridden by the main loop for comparison)
    * `PUBLIC_CACHE_FRACTION` (for "partial_public_private")
    * `PREDICTION_INTERVAL` (for predictive caching)
    * `ADJACENCY_MATRIX` (defines edge network connections)
    * `TOTAL_WORKFLOW_SUBMISSIONS` (total workflow instances to run)
    * `MIN_WORKFLOW_SUBMISSION_INTERVAL`, `MAX_WORKFLOW_SUBMISSION_INTERVAL` (inter-arrival times)
    * `PEGASUS_WORKFLOW_FILEPATHS` (paths to your DAX XML files)

2.  **Execute the Simulator:**
    Ensure your virtual environment is active and you are in the project's root directory (`faas-edge-cloud-sim/`). Then run:
    ```bash
    python -m src.simulator
    ```

## Interpreting Results

The simulation will print statistics to the console for each cache sharing policy run. Additionally, it will generate several plots in the `logs/` directory at the project root:

-   `policy_comparison_workflow_completion.png`: Bar chart showing percentage of workflows completed on time for each cache policy.
-   `policy_comparison_edge_ratio.png`: Bar chart showing the percentage of tasks executed on edge nodes for each cache policy.
-   `policy_comparison_cold_starts.png`: Bar chart comparing total cold starts across different cache policies.
-   `policy_comparison_avg_edge_utilization.png`: Bar chart showing average edge node utilization for each cache policy.
-   *(Optional individual run plots, if re-enabled in `simulator.py`'s `run_simulation` method):*
    -   `workflow_completion_status.png`
    -   `edge_cloud_task_ratio.png`
    -   `edge_utilization.png`

Analyze these metrics and plots to understand the impact of different scheduling and caching policies on overall system performance, task execution, and resource utilization.

## Future Enhancements (Ideas for further development)
-   More sophisticated load balancing techniques.
-   Dynamic scaling of edge node resources.
-   Advanced predictive models for caching (e.g., machine learning-based).
-   Modeling network congestion and varying inter-node latencies.
-   Heterogeneous edge nodes with different capacities and processing rates.
-   User-defined QoS requirements per workflow/task.

---