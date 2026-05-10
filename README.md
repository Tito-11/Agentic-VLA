# Agentic-VLA

**Agentic-VLA: A Reflective Multi-Agent Architecture for Enhancing VLA Models in Long-Horizon Tasks**

---

## Overview

This repository contains the code and experimental framework for **Agentic-VLA** (formerly LAMA-VLM / Agentic RAG-VLM), a unified framework for robotic manipulation that bridges semantic understanding and physical execution. 

To address the limitations of pure end-to-end VLA models (which lack long-horizon planning and fault reflection), we have upgraded the system to **Agentic-VLA**. This architecture integrates **Retrieval-Augmented Generation (RAG)**, **Vision-Language Models (VLMs)**, **lifelong learning**, and **multi-agent state orchestration** (System 2) as a robust wrapper around high-frequency VLA execution experts (System 1).

<p align="center">
  <img src="paper/figures/fig.1.png" alt="System Overview" width="800"/>
</p>

### Agentic-VLA Architecture (Heterogeneous Dual-System)

Our framework operates on a **Dual-Model Architecture**, seamlessly bridging a "Slow Thinking" pure VLM brain with a "Fast Thinking" VLA cerebellum:

1. **System 2 Thinking Layer (External VLM Brain / Qwen-VL Max):** A pure, highly-capable Vision-Language Model acts as the cognitive brain. Untainted by robot action-finetuning, it retains profound spatial and logical reasoning. Using Topology-Aware Graph RAG, it analyzes the current perspective, extracting `supports` and `occluded_by` relations to generate macro instructions, safe anti-collision yaw angles, and explicit state transitions.
2. **Visual Prompting & Masking (Inspired by $VLA^2$):** The VLM Brain generates environmental parsing masks (e.g., bounding boxes or transparent occlusion masks) to eliminate texture biases and enable the underlying VLA to handle Out-of-Distribution (OOD) unseen concepts.
3. **System 1 Execution Layer (VLA Cerebellum / Frozen pi0):** The semantic instructions and visual prompts are routed to the frozen `pi0` foundation model. Operating strictly as a "cerebellum", `pi0` focuses purely on decoding these macro-commands into continuous 6-DoF Action Chunks to directly drive the robot arms with high precision.
4. **Reflective Loop (Critic Agent & Evo-KAM):** Post-execution, the VLM Brain evaluates execution states via visual and force feedback. If an error occurs (e.g., collision, slip), it intercepts the failure, updates ChromaDB memory via Evolutionary Knowledge Memory (Evo-KAM), and triggers re-planning—achieving a 100% recovery rate on soft failures that pure VLAs cannot handle.

### Key Components

1. **Topo-Graph RAG (Topological-Aware Graph RAG):** Eliminates target-centric retrieval by extracting subgraphs containing `supports`, `occluded_by`, and `next_to` edges. Enables semantic subgraph isomorphism matching in a high-dimensional vector space.
2. **Evo-KAM (Evolutionary Knowledge & Affordance Memory):** A lifelong memory system powered by ChromaDB. Uses a conflict threshold mechanism to dynamically decay reliability weights and evolve prescriptive actions based on VLM reflections.
3. **V-PCC (Visual-Progressive Context Compression):** A micro-compact mechanism inspired by SnapKV, dynamically discarding redundant image frames and condensing historical text logs to prevent context explosion during long-horizon manipulation.
4. **DAG-EAM (Directed Acyclic Graph Explicit Action Modeling):** Integrates LangGraph to orchestrate explicit state transitions between specialized expert agents (`Vision`, `Plan`, `Execute`, `Critic`), allowing safe conditional routing, retroactive backtracking, and persistent action snapshots.

---

## Project Structure

```
Agentic-VLA/
├── robot_grasp_rag/          # Core framework for autonomous manipulation
│   ├── agent/                # Multi-Agent StateGraph pipeline
│   │   └── optimized_multi_agent.py
│   ├── core/                 # Knowledge base & RAG implementations
│   │   ├── graph_rag.py
│   │   └── memory_and_context.py
│   ├── knowledge_base/       # ChromaDB Vector store & grasp memory
│   │   └── vector_store.py
│   ├── vla_model/            # Frozen VLA integration (OpenPI/Pi0)
│   │   └── pi0_executor.py
│   ├── utils/                # Utilities (pose representation, logging)
│   ├── config/config.yaml    # System configuration for VLM and SAPIEN
│   ├── scripts/              # Evaluation & benchmarking scripts
│   │   └── run_lama_benchmark.py
│   └── run_optimized_simulation.py # Main deployment simulation loop
├── quantization/             # VLM quantization tools (FP8/INT4 capabilities)
├── results/                  # Generated benchmark tables & execution traces
├── paper/                    # LaTeX source & PAPER_FRAMEWORK
├── README.md                 # System overview and deployment guidelines
└── LICENSE
```

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 12.0
- PyTorch >= 2.0

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Agentic-VLA.git
cd Agentic-VLA

# Install dependencies (requires LangGraph and ChromaDB capabilities)
pip install -r robot_grasp_rag/requirements.txt

# The project uses OpenPI (physical-intelligence/pi0) base model
# Use the automated script to download the official pi0_base weights via Google Cloud Storage
python scripts/download_pi0.py
# Weights will be automatically configured to: weights/openpi-assets/checkpoints/pi0_base
```

### Configuration

Edit `robot_grasp_rag/config/config.yaml` to set your model path:

```yaml
vlm:
  model_path: "/path/to/your/Qwen3-VL-8B"
```

---

## Usage

### Run Full Experiments

```bash
# Run the Interactive Agentic-VLA Framework (Multi-Agent StateGraph)
LD_PRELOAD=/home/fudan222/miniconda3/envs/roboagent/lib/libstdc++.so.6 \
python -m robot_grasp_rag.run_optimized_simulation
```

### Run Agentic-VLA Ablation Benchmarks (LIBERO)

Validates Agentic-VLA across extreme long-horizon tracks, injecting state-gap collisions and OOD concepts.

```bash
# Execute the comprehensive benchmarking suite
conda run -n roboagent python scripts/run_agentic_vla_libero.py
```

---

## Experimental Results (LIBERO Benchmark 2026-05)

Our method is evaluated on the LIBERO long-horizon multi-task benchmark. To demonstrate the necessity of our System 2 (Agentic) architecture, we conduct rigorous ablation studies against the state-of-the-art pure end-to-end VLA model (pi0).

| Model Configuration | Task Success Rate | Recovery Rate |
| :--- | :---: | :---: |
| **Base VLA (pi0)** | 10.0% | 0.0% (No Critic) |
| **pi0 + Vision Agent** | 15.0% | 0.0% |
| **pi0 + Vision + Planner Agent** | 75.0% | 0.0% |
| **Agentic-VLA (pi0 + Full Agentic Framework)** | **85.0%** | **100.0% (Full Loop Recovery)** |

**Key Takeaways:**
- **Planner Agent Lift (+70%):** Eliminates State Gap collisions between atomic tasks by generating smooth transitional actions.
- **Critic Agent Lift (+10% SR, 100% Recovery):** Converts soft physical failures (slips, non-perfect grasps) into successful task completion via self-reflective backtracking.

Detailed results including OOD generalization and latency analysis are available in `results/iros2026/` and `paper/PAPER_FRAMEWORK.md`.

---

## Changelog & Architecture Evolution

This project has undergone significant evolutions from a standard RAG-VLM script to a fully distributed, VLA-integrated multi-agent system:

1. **Real Physical Execution with OpenPI (2026-05):**
   - Transferred the framework to run directly inside the real Mujoco/Robosuite physics engine provided by the LIBERO benchmark.
   - Built a dedicated `openpi_env` to load the official `physical-intelligence/pi0_base` weights (11.2 GiB), eliminating mock execution and outputting actual robotic Action Chunks.
   - End-to-end framework verification generates authentic ablation videos saved under `results/videos/`.

2. **Agentic-VLA Framework Integration (2026-05):**
   - **OpenPI / pi0 Execution:** Replaced traditional IK-driven execution with continuous Action Chunks generated by VLA.
   - **$VLA^2$ Visual Prompting:** Empowered the Vision Agent to generate Bounding Box and transparent masks to guide the frozen VLA base, eliminating dependency on target texture.
   - **Sci-VLA Long-Horizon Bridging:** Added `node_transition_agent` in LangGraph to generate safe intermediate postures between atomic tasks, preventing cascade collisions in sequential setups.
   
2. **Robust Multi-Agent Workflow (2026-04):**
   - Evolved into a **MUSE/OpenClaw-like architecture** featuring `Vision Agent`, `Planning-Execution Agent`, and `Critic Agent`.
   - Tool calls are wrapped with sub-routines requiring explicit `reason` parameters ("Why did I do this?"), drastically improving interpretability for the Critic Agent.

3. **Lifelong Memory & Context Compression:**
   - **ChromaDB Vector Store** sync in `LifelongMemoryManager` automatically updates grasp parameters based on open-vocabulary reflection when failure thresholds are hit.
   - **Progressive Context Compression** module prevents Token explosion via textual sequence summaries ("Micro-Compact context").

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{agentic-vla2026,
  title={Agentic-VLA: A Reflective Multi-Agent Architecture for Enhancing VLA Models in Long-Horizon Tasks},
  author={Anonymous},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project builds upon [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) for vision-language understanding and the [OpenPI](https://github.com/Physical-Intelligence/openpi) model for action generation. We thank their respective teams for their open-source contributions.
