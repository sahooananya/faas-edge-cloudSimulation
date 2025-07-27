# FaaS Edge-Cloud Simulator

## Overview

This project is a **Discrete Event Simulator** for modeling and analyzing **Function-as-a-Service (FaaS)** deployments across a **heterogeneous Edge-Cloud continuum**. It enables researchers to simulate serverless function scheduling, caching strategies, and network dynamics under various configurations. The simulator is designed to support both theoretical exploration and practical experimentation in **edge computing, cloud offloading, and resource orchestration**.

This tool is especially tailored for researchers and PhD students exploring:

- Edge-first scheduling and hybrid architectures  
- Predictive and cooperative caching strategies  
- Task dependency resolution in scientific workflows  
- Performance trade-offs in latency vs resource utilization

---

## Features

### ✅ Core Simulation Framework

- **Discrete Event Engine** with event queue for time-ordered execution  
- Models for key entities: `Task`, `Workflow`, `EdgeNode`, `Cloud`, `CacheManager`  
- **Workflow parsing** from Pegasus DAX XML for realistic DAGs  
- **Initial Greedy & EDF Scheduling**

### ⚙️ Enhanced Modules

- **Run-time Statistics**: task turnaround time, edge utilization, task deadlines  
- **Scheduling Policies**:  
  - *Earliest Deadline First (EDF)*  
  - *Critical Path First (CPF)* (optional)  
- **Caching Policies**:
  - *Full Public Cache*
  - *Partial Public-Private Cache*
  - *Full Private Cache*
- **Plotting Support**: Comparative graphs of task distribution, deadline misses, utilization  
- **Batch Evaluation Script**: `run_multiple.py` automates experiments across combinations

---

## Project Structure

```

faas-edge-cloudSimulation/
│
├── main.py                      # Entry script (can be extended)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── src/
│   ├── config.py                # Configurable simulation parameters
│   ├── simulator.py             # Core simulation engine
│   ├── scheduler.py             # Scheduling policies
│   ├── cache\_manager.py         # Caching strategies
│   ├── edge\_network.py          # Edge node and cloud modeling
│   ├── models.py                # Task, Workflow, Node definitions
│   ├── parser.py                # DAX XML workflow parser
│   ├── run\_multiple.py          # Script to compare multiple setups
│   └── results/plotting.py      # Plot generation and visualization

````

---

## Installation

### 🔧 Prerequisites

- Python ≥ 3.8
- pip

### 💻 Setup

```bash
git clone https://github.com/sahooananya/faas-edge-cloudSimulation.git
cd faas-edge-cloudSimulation
pip install -r requirements.txt
````

---

## Usage

### 🔁 Run a Single Simulation

```bash
python src/simulator.py
```

This uses the default configuration and policies in `src/config.py`.

### 🧪 Run Multiple Experiments

```bash
python src/run_multiple.py
```

This will automatically evaluate:

* EDF + \[Full Public, Partial Public-Private, Full Private] caches
* Collect results, generate comparison plots

---

## Configuration

All simulation parameters are in `src/config.py`:

```python
NUM_TASKS = 50
EDGE_NODES = 3
SCHEDULING_POLICY = "EDF"  # or "CPF"
CACHE_POLICY = "partial_public_private"
WORKFLOW_FILE = "sample_workflow.xml"
```

Modify these to match your experiment's scale and policy choice.

---

## Input Format

Workflows must be in [Pegasus DAX XML](https://pegasus.isi.edu/documentation/) format.
Sample files and generator tools can be added for convenience.

---

## Output

After simulation:

* Task status logs
* Edge/Cloud utilization
* Deadline hit/miss stats
* Graphs saved in `results/`

Sample Plots:

* Deadline miss comparison
* Edge utilization %
* Cache hit rates (if enabled)

---

## Research Applications

This simulator supports exploration in:

* **Adaptive edge-cloud scheduling**
* **Predictive and cooperative caching**
* **Latency-constrained computing**
* **Edge-aware scientific workflow execution**
* **Energy-aware offloading strategies**

You may extend modules to:

* Integrate real network latency traces
* Add reinforcement learning-based policies
* Simulate heterogeneous edge nodes

---

## Citation & Credits

If used in a research paper, please credit:

```
Ananya Sahoo, "FaaS Edge-Cloud Simulation Framework", 2025.
```

Due acknowledgment will be appreciated. This work is open to collaborative research and improvement.

---

## Contact

For collaboration or queries, feel free to reach out to:

**Ananya Sahoo**
B.Tech, Computer Science
KIIT University
📧 [sahooananya036@gmail.com](mailto:sahooananya036@gmail.com) 

