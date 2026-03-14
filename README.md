## CA-DDQN: Channel Allocation in Satellite IoT with CA-DRL and Double DQN

This repository implements and extends the CA-DRL (Channel Allocation using Deep Reinforcement Learning) method from the paper
“Dynamic Channel Allocation for Satellite Internet of Things via Deep Reinforcement Learning” by Liu et al.

Phase 1 reproduces CA-DRL with a DQN agent; Phase 2 replaces the learning algorithm with Double DQN (DDQN) while keeping the
same environment, state, action, and reward definitions.

### Project structure

- **environment**: satellite IoT simulation
  - `wireless_env.py`: CA-DRL single-satellite environment (state and reward as in the paper)
  - `traffic_model.py`: short-burst random traffic model
- **algorithms**:
  - `cadrl_agent.py`: DQN-based CA-DRL agent and trainer
  - `ddqn_agent.py`: DDQN agent and trainer
- **training**:
  - `train_cadrl.py`: CA-DRL reproduction training script
  - `train_ddqn.py`: DDQN training and comparison script
- **evaluation**:
  - `metrics.py`: episode-level metrics and baseline comparison helpers
  - `plots.py`: training curves and algorithm comparison plots
- **configs**:
  - `experiment_config.yaml`: central experiment configuration (environment, training, baselines)
- **utils**:
  - `config_loader.py`: typed access to the YAML config
  - `reproducibility.py`: seeding and deterministic settings
- **results**:
  - `ca-drl/`: CA-DRL reproduction outputs
  - `ddqn/`: DDQN experiment outputs
- **models**:
  - `ca_drl_model.pt`: trained CA-DRL DQN agent
  - `ddqn_model.pt`: trained DDQN agent

### Environment and algorithm

The environment matches the CA-DRL paper:

- **Channels**: 14 uplink channels (2 kHz each)
- **Traffic**: random short-burst transmissions with maximum duration 15 time units
- **State** \(Eq. (3)\):  
  \( S_t = [\text{tasks}_1..T, \text{bw}_1..bw_N, q_1..q_N] \)  
  where:
  - `tasks_i`: number of active transmissions in timestep \(i\) over a window of `state_window_size` steps
  - `bw_i`: data size requirement (bits) of waiting sensor \(i\)
  - `q_i`: duration requirement (time units) of waiting sensor \(i\)
- **Reward** \(Eq. (4)\):  
  \( R_t = - \sum_x q_x \) over all tasks in the waiting queue (implemented exactly as negative sum of durations).

CA-DRL uses a DQN with an MLP over this state; DDQN uses the same architecture but Double Q-learning targets.

### Installation

From the project root (`ca-ddqn`):

```bash
python -m venv env
source env/bin/activate  # on Windows: env\\Scripts\\activate
pip install -r requirements.txt
```

### Phase 1 – Reproduce CA-DRL

Run CA-DRL training (DQN) using the paper’s environment and reward:

```bash
cd ca-ddqn
python -m training.train_cadrl
```

Outputs:

- `results/ca-drl/ca_drl_training_results.json`: summary metrics (average delay, reward, throughput)
- `results/ca-drl/plots/cadrl_training_rewards.png`: reward vs episode
- `results/ca-drl/plots/cadrl_training_delays.png`: delay vs episode
- `models/ca_drl_model.pt`: trained CA-DRL DQN model checkpoint

For comparison with baselines:

- FCFS and Random Access are **not simulated**; instead, their average delay and reward are taken directly from Table 4 of the paper:
  - FCFS: delay 1.08, reward -773.1
  - Random Access: delay 1.25, reward -910.2

### Phase 2 – DDQN enhancement

Train the DDQN-based algorithm on exactly the same environment and reward:

```bash
cd ca-ddqn
python -m training.train_ddqn
```

Outputs:

- `results/ddqn/ddqn_training_results.json`: DDQN summary metrics and algorithm comparison table
- `results/ddqn/plots/ddqn_training_rewards.png`, `ddqn_training_delays.png`: DDQN training curves
- `results/ddqn/plots/algorithm_comparison_delay.png`, `algorithm_comparison_reward.png`: bar plots comparing
  - FCFS (paper baseline)
  - Random Access (paper baseline)
  - CA-DRL (reproduced)
  - DDQN (proposed)

- `models/ddqn_model.pt`: trained DDQN model checkpoint

### Evaluation from saved models (validation)

After training and saving checkpoints, you can run **pure evaluation** (no learning) using:

- **Evaluate CA-DRL model vs baselines**:

  ```bash
  cd ca-ddqn
  python -m evaluation.evaluate_cadrl_model
  ```

  Outputs under `results/model_eval/ca_drl/`:
  - `ca_drl_model_evaluation.json`: CA-DRL summary metrics and comparison vs FCFS/Random
  - `algorithm_comparison_cadrl_model.png`: bar plot for CA-DRL vs baselines

- **Evaluate DDQN model vs baselines**:

  ```bash
  cd ca-ddqn
  python -m evaluation.evaluate_ddqn_model
  ```

  Outputs under `results/model_eval/ddqn/`:
  - `ddqn_model_evaluation.json`: DDQN summary metrics and comparison vs baselines
  - `algorithm_comparison_ddqn_model.png`: bar plot for DDQN vs baselines

- **Evaluate CA-DRL and DDQN models together (head-to-head)**:

  ```bash
  cd ca-ddqn
  python -m evaluation.evaluate_saved_models
  ```

  Outputs under `results/model_eval/`:
  - `model_evaluation_results.json`: CA-DRL vs DDQN vs baselines
  - `algorithm_comparison_models.png`: bar plot comparing all algorithms

### Reproducibility

- All experiments use a single YAML config at `configs/experiment_config.yaml` for:
  - satellite and environment parameters
  - training hyperparameters
  - evaluation settings
  - fixed FCFS and Random baselines from the paper
- `utils/reproducibility.py` seeds Python, NumPy and PyTorch and enables deterministic behaviour where supported.

### Notebooks

To explore results interactively, create or open:

- `notebooks/cadrl_reproduction.ipynb`: runs CA-DRL training and visualises reward/delay/throughput.
- `notebooks/ddqn_experiment.ipynb`: runs DDQN training, then loads the JSON summaries and reproduces comparison plots.

Both notebooks should import the same modules used by `training/train_cadrl.py` and `training/train_ddqn.py` so that
script and notebook experiments stay consistent.
