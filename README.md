## ELEC-E8125 Reinforcement Learning – Exercises & Project

English summary of the repository structure and how to run the materials.

### Structure
- `ex1`–`ex6`: Course exercises (notebooks plus helper scripts, configs, and saved results).
- `project`: Final project with PPO/DDPG implementations, configs, and result artifacts.
- Each `ex*/cfg` contains YAML configs; `imgs/`, `results/`, and `video/` hold outputs from prior runs.

### Getting started
1. Python 3.10+ recommended. Create a virtual env (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install requirements used across notebooks/scripts (adjust as needed per exercise):
   ```bash
   pip install torch torchvision torchaudio gymnasium[box2d] numpy matplotlib pandas pyyaml tqdm
   ```
   Some tasks may also need `imageio`, `moviepy`, or `jupyter`.
3. Launch notebooks: `jupyter lab` (or `jupyter notebook`) and open the relevant `ex*.ipynb` or `project.ipynb`.

### Running scripts (examples)
- Many exercises expose training entrypoints like `train.py` with YAML configs:
  ```bash
  python ex1/train.py --config ex1/cfg/cartpole_v1.yaml
  python ex4/train.py --config ex4/cfg/cartpole_dqn.yaml
  ```
- Configs can be tweaked to switch environments, seeds, or model settings.

### Tips
- Use GPU if available for heavier tasks (DDPG/PPO), otherwise expect longer runs on CPU.
- Existing `results/` and `video/` outputs illustrate expected behavior; regenerate them by re-running the scripts with the same configs.
- Keep large artifacts (videos, model checkpoints) out of commits unless necessary.

### Contribution workflow
- Create a clean branch, make changes, run/record results, then commit and push.
- For notebooks, clear or minimize unnecessary outputs before committing to reduce repo size.

