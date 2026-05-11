# Albert Dog Robot — PPO Reinforcement Learning

Train a simulated 4-legged robot (Albert) to walk using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) in MuJoCo. The trained policy is small enough to run on an ESP32 microcontroller.

## Overview

The actor network outputs **ΔΔθ** actions — changes to a persistent delta buffer — implementing acceleration-level joint control with structural smoothness by design.

```
State  (24): jpos(8) + jvel(8) + delta(8)
Action  (8): ΔΔθ_i  →  delta  →  joint targets
Actor arch:  24 → ReLU(64) → tanh(8)   (~8 KB on ESP32)
```

**Target speed:** 0.125 m/s (1 m in 8 s)

## Files

```
AllLegs_def.ipynb  — main training notebook (PPO + GAE + MuJoCo)
dog.xml            — MuJoCo robot model (8 joints, 4 legs)
models/            — saved checkpoints (.pth) and policy.h (C header)
trajectories/      — exported joint trajectories (.npy, .csv, .h)
movies/            — rendered evaluation videos (.mp4)
plots/             — training curves
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate albert-rl
jupyter notebook AllLegs_def.ipynb
```

Open `AllLegs_def.ipynb` and run all cells. Training runs for `N_EP = 5000` episodes by default. Checkpoints are saved every 50 episodes to `models/`.

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `N_EP` | 5000 | Training episodes |
| `HIDDEN_DIM` | 64 | Actor hidden layer size |
| `TARGET_VX` | 0.125 m/s | Forward speed target |
| `MAX_DDELTA` | 0.02 rad/step | Joint acceleration limit |
| `DELTA_LIMIT` | 0.05 rad/step | Delta buffer cap |
| `LR` | 3e-4 | Adam learning rate |
| `GAE_LAMBDA` | 0.95 | GAE trace-decay for advantage estimation |
| `GAMMA` | 0.99 | Discount factor |
| `N_STEPS_PER_UPDATE` | 4096 | PPO rollout buffer size |
| `PPO_EPOCHS` | 10 | Gradient epochs per update |

## Reward Function

```
R = -VX_TRACK_W * (vx - TARGET_VX)²   # velocity tracking
  + ALIVE_BONUS                         # stay upright
  - 0.1 * |vy|                         # no lateral drift
  - ACTION_REG_W * ||ΔΔθ||²            # smooth accelerations
  - DELTA_REG_W  * ||delta||²          # prevent delta drift
  + JVEL_W  * jvel_mag                 # reward joint activity
  + JEXC_W  * excursion               # reward joint excursion
  - YAW_W   * Δyaw²                   # stay straight
```

Episodes terminate early if the robot falls (body height < 3 cm or pitch > 45°).

## Advantage Estimation

The definitive version uses **Generalized Advantage Estimation (GAE)** (`GAE_LAMBDA = 0.95`) instead of plain TD(0). GAE interpolates between high-variance Monte Carlo returns and low-variance (but biased) TD(1), giving more stable policy gradients over long episodes.

## Resume / Eval

To resume training from a checkpoint, set in the notebook:

```python
LOAD_MODEL = "models/nn_ppo_ddtheta_ep999.pth"
```

To run evaluation only (deterministic rollout + video + trajectory export):

```python
LOAD_MODEL = "models/nn_ppo_ddtheta_best.pth"
EVAL_ONLY  = True
```

## ESP32 Export

After training, the notebook exports:

- `models/policy.h` — actor weights + `policy_step()` C function, ready to `#include`
- `trajectories/best.h` — full 800-step trajectory as `int16_t` array (12.5 KB)
- `trajectories/one_cycle.h` — single gait cycle extracted via FFT (19 steps at 5.25 Hz, ~0.3 KB); preferred for looping on the robot

The notebook auto-detects the dominant gait frequency with `np.fft.rfft` and prefers `one_cycle.h` in the ESP32 bundle if a valid cycle is found.

Drop both into your Arduino sketch folder:

```
albert_firmware/
  albert_firmware.ino
  policy.h          ← neural network inference
albert_trajectory/
  albert_trajectory.ino
  trajectory.h      ← pre-recorded open-loop trajectory
```

## Dependencies

See `environment.yml`. Core requirements:

- Python 3.10+
- PyTorch 2.x (CPU sufficient for this network size)
- MuJoCo 3.x + `mujoco` Python bindings
- `mediapy` for video rendering
- `matplotlib`, `numpy`, `tqdm`
