# Multi-Agent Social Navigation & Safety-Aware Crowd Navigation

> An RL-based crowd navigation framework that achieves robust, 
generalizable safe navigation through uncertainty quantification 
and constrained policy learning — validated in simulation and 
deployed on a real mobile robot.

---

# Note
This project was conducted as part of TASL Lab at university of California, Riverside. The codebase, robot hardware, and computing infrastructure were owned by the lab. Accordingly, no source code or experimental data are published in this repository. This README documents the system design, technical architecture, and implementation approach for reference and portfolio purposes.

# Personal Reimplementation & Planned Extensions
In Progress: A personal reimplementation of this system in a simulated environment is currently in progress. Since the original codebase belongs to the university research lab, this reimplementation is being developed independently from scratch. Two extensions beyond the original architecture are being incorporated, each targeting a distinct layer of the system.

---

## Project Overview

Safely navigating among crowds is one of the core challenges 
in deploying mobile robots in human-populated environments — 
hospitals, warehouses, public spaces. The naive approach of 
treating humans as moving obstacles fails because human motion 
is intentional, social, and group-structured.

This project addresses that challenge through a system that 
combines:

- **Multi-agent trajectory prediction** — forecasting where 
each pedestrian is likely to move, using attention-based graph 
networks that model both pairwise and group-wise interactions
- **Uncertainty quantification** — wrapping those predictions 
with statistically valid uncertainty bounds using Adaptive 
Conformal Inference (ACI), so the robot knows *how much to 
trust* each prediction
- **Constrained reinforcement learning** — a policy that 
explicitly accounts for prediction uncertainty in its cost 
function, learning to stay safe even when predictions are 
wrong or crowd dynamics shift

The result is a navigation system that generalises across 
out-of-distribution scenarios — faster pedestrians, different 
crowd behavior models, pedestrian group dynamics — where 
standard RL-based navigation methods degrade significantly.

The learned policy was deployed on a physical Mecanum-wheel 
robot in real outdoor crowd environments without any 
fine-tuning, demonstrating direct sim-to-real transfer.

---

## Core System — Built by the Lab

> The following components form the core of the project. 
They were designed and implemented by PhD researchers at TASL. 
They are documented here to provide full context for the 
project. My specific contributions are described separately 
in the [My Contributions](#my-contributions) section.

### Trajectory Prediction

The prediction module forecasts future pedestrian positions 
using two models:

- **Constant Velocity (CV)** — rule-based linear extrapolation 
from current velocity; fast and interpretable
- **Gumbel Social Transformer (GST)** — a learning-based model 
that encodes sparse pedestrian interactions and handles 
partially-detected agents in dense crowds

Both predictors output K-step future positions 
$\hat{p}_{h,k}(t)$ for each pedestrian $h$.

### Adaptive Conformal Inference (ACI) for Uncertainty

Raw trajectory predictions are wrapped with uncertainty bounds 
using **DtACI** (Dynamically-tuned Adaptive Conformal 
Inference) — an online, distribution-free uncertainty 
quantification method that adapts as crowd dynamics shift.

For each pedestrian $h$ and prediction step $k$, the actual 
prediction error is computed as:

$$\delta_{h,k}(t) = \| p_h(t) - \hat{p}_{h,k}(t-k) \|_2$$

The estimated uncertainty $\hat{\delta}_{h,k}$ is updated 
online across $M$ parallel estimators with different learning 
rates $\gamma^{(m)}$:

$$\hat{\delta}^{(m)}_{h,k}(t) = \hat{\delta}^{(m)}_{h,k}(t-1) 
- \gamma^{(m)} \left(\alpha - \text{err}^{(m)}_{h,k}(t)\right)$$

where $\alpha$ is the coverage parameter (set to 0.1 for 90% 
coverage) and $\text{err}^{(m)}_{h,k}(t) = 1$ if the 
estimated error undershot the actual error, else 0.

The best estimator is selected at each step via exponential 
weighting using the **pinball loss**:

$$\ell(\delta, \hat{\delta}) = 
\begin{cases} 
\alpha(\delta - \hat{\delta}) & \text{if } \delta \geq 
\hat{\delta} \\ 
(\alpha - 1)(\delta - \hat{\delta}) & \text{if } \delta < 
\hat{\delta}
\end{cases}$$

This produces a per-pedestrian, per-step uncertainty radius 
$\hat{\delta}_{h,k}$ that expands automatically when crowd 
dynamics shift — giving the robot a larger safety margin 
precisely when predictions become less reliable.

### Safety-Critical Area Design

Each pedestrian's safety zone is defined as a union of two 
sub-areas:

$$D_1(p_\text{ego}) = \{ p_\text{ego} : \|p_\text{ego} - 
p_h\| \leq r_\text{ego} + r_h + r_\text{comfort} \}$$

$$D_2(p_\text{ego}) = \{ p_\text{ego} : \|p_\text{ego} - 
\hat{p}_{h,k}\| \leq r_\text{ego} + r_h + \hat{\delta}_{h,k} 
\}$$

- $D_1$: fixed comfort zone around the current pedestrian 
position
- $D_2$: uncertainty-scaled zone around each predicted 
future position — **grows dynamically** with prediction error

The robot incurs a cost $d_\text{intr,t}$ equal to the 
maximum intrusion depth into any pedestrian's safety zone 
at each timestep.

### Constrained Reinforcement Learning (CRL)

Navigation is formulated as a **Constrained Markov Decision 
Process (CMDP)**. The objective is:

$$\max_\pi \; \mathbb{E}_\pi \left[ \sum_{t=0}^{T} R_t 
\right] \quad \text{subject to} \quad \mathbb{E}_\pi 
\left[ \sum_{t=0}^{T} d_{\text{intr},t} \right] \leq 
\tilde{d}$$

where $\tilde{d}$ is a predefined intrusion budget. The 
cost at each step is:

$$C_t(S_t, A_t) = \mu \cdot d_{\text{intr},t}$$

Optimization uses **PPO-Lagrangian**, which maintains a 
Lagrange multiplier $\lambda$ to balance reward maximization 
against the cost constraint. Two separate critics are 
maintained — one for reward value $V^R$, one for cost value 
$V^C$. The combined advantage for the actor is:

$$\hat{A}'_t = \frac{\hat{A}^R_t - \lambda \hat{A}^C_t}
{1 + \lambda}$$

$\lambda$ is updated via gradient descent:

$$\ell^\lambda_t = -\lambda \left(\bar{C} - \tilde{d}_C\right)$$

### Policy Network

The policy network processes uncertainty-augmented 
observations through:

1. **H-H Attention** — models human-human interactions via 
multi-head attention over all pedestrian pairs
2. **H-R Attention** — fuses robot state into the 
human-centric embeddings
3. **GRU** — captures temporal dynamics across timesteps
4. **Actor + Reward Critic** (shared backbone) + 
**Cost Critic** (separate network)

The prediction uncertainty $\hat{\delta}_{h,k}$ is 
concatenated with trajectory predictions before entering 
the attention layers — making the robot's policy explicitly 
uncertainty-aware at the representation level.

---

## My Contributions

My involvement focused on extending the system's perception 
and environment mapping capabilities — the infrastructure 
layer the RL policy runs on top of.

### 1. LiDAR-Camera Sensor Fusion & 3D Perception

The baseline system relied on a single **2D LiDAR** 
(RPLIDAR-A1) operating on one horizontal scan plane. This 
limits detection robustness — pedestrians partially outside 
the scan plane, children, or agents at varying heights can 
be missed entirely.

**What I contributed:**

- Contributed to extending the perception pipeline to 
incorporate **RGB-D camera** data alongside the LiDAR 
input, enabling multi-modal sensor fusion for improved 
human detection
- Supported the transition to a **3D LiDAR-compatible** 
perception front-end to capture richer geometric context 
around pedestrians across multiple elevation planes
- Worked on improving detection robustness in scenarios 
with partial occlusions and varied pedestrian heights
- Validated detection improvements through comparative 
testing in simulation

**Hardware used:**

- 3D LiDAR for volumetric point cloud capture
- RGB-D Camera for depth-fused visual detection
- Sensor fusion pipeline replacing the original 2D scan-only 
approach

### 2. SLAM-Based Environmental Mapping

The original system modeled static obstacles only as 
circular agents — a significant simplification that fails 
in real environments containing walls, corridors, and 
arbitrarily shaped structures.

**What I contributed:**

- Contributed to integrating a **SLAM-based mapping 
module** to build and maintain live occupancy grid maps 
of the navigation environment
- Supported modification of the robot's observation space 
to incorporate map-based obstacle representations alongside 
the RL policy input
- Assisted in testing the extended system in structured 
simulation environments beyond the open-area scenarios 
used in baseline training
- Validated navigation robustness in scenarios with 
non-circular static obstacles including walls and corridor 
geometry

### 3. System Validation & Deployment Support

- Contributed to end-to-end system validation through 
simulation runs and real-world experiments on the mobile 
robot platform
- Assisted in evaluating navigation safety and robustness 
metrics across multiple crowd scenarios

---

## Hardware Stack

| Component | Model / Details |
|-----------|----------------|
| Robot Platform | ROSMASTER X3 (Mecanum-wheel mobile robot) |
| Onboard Compute | Laptop with NVIDIA RTX 3070 (Mobile) — connected via router |
| 2D LiDAR (baseline) | RPLIDAR-A1 — 360°, ~6 Hz scan frequency |
| 3D LiDAR (extended) | 3D LiDAR sensor — volumetric point cloud perception |
| RGB-D Camera | RGB-D camera — depth-fused visual detection |
| Locomotion | 4× Mecanum wheels — holonomic motion (independent vx, vy control) |
| Middleware | ROS 2 |

---

## Software Stack

| Category | Tools |
|----------|-------|
| Middleware | ROS 2 |
| Simulation | CrowdNav (RL training), Gazebo |
| Deep Learning | PyTorch |
| RL Framework | PPO-Lagrangian (OmniSafe) |
| Trajectory Prediction | Gumbel Social Transformer (GST) |
| Uncertainty Quantification | DtACI (Dynamically-tuned Adaptive Conformal Inference) |
| Human Detection | DR-SPAAM (2D range-based learning detector) |
| Multi-Object Tracking | SORT (Simple Online and Realtime Tracking) |
| SLAM | SLAM Toolbox / Cartographer (ROS 2) |
| Languages | Python, C++ |

---

## Planned Improvements

These improvements are planned for my independent 
simulation reimplementation of this project in 
ROS 2 + Gazebo.

### Improvement 1: Density-Adaptive Uncertainty Coverage

**Problem with the current approach:**

The ACI module uses a fixed coverage parameter 
$\alpha = 0.1$ (90% coverage) regardless of crowd 
density. In sparse environments this is unnecessarily 
conservative — slowing the robot without safety benefit. 
In highly dense environments it may still be insufficient.

**Planned modification:**

Implement a dynamic $\alpha$ schedule driven by a 
real-time local density estimate $\rho(t)$:

$$\alpha(t) = \alpha_{\min} + (\alpha_{\max} - 
\alpha_{\min}) \cdot f(\rho(t))$$

where $\rho(t)$ is estimated as the number of tracked 
pedestrians within a fixed radius $r_\rho$ of the robot:

$$\rho(t) = \left| \{ h : \|p_h(t) - p_\text{ego}(t)\| 
\leq r_\rho \} \right|$$

and $f(\cdot)$ is a monotonically decreasing function — 
higher density → lower $\alpha$ → tighter (more 
conservative) uncertainty bounds.

**Expected outcome:**

- Reduced unnecessary slow-downs in low-density scenarios
- Maintained safety guarantees in high-density scenarios
- More natural, human-like navigation pacing across 
varied environments

---

### Improvement 2: Group-Density Aware Cost Shaping

**Problem with the current approach:**

The CRL cost function penalises intrusion into individual 
pedestrian uncertainty areas — but treats navigating near 
a single pedestrian identically to entering the middle of 
a cohesive group. Entering a group space is more socially 
disruptive, harder to recover from, and can trap the robot.

**Planned modification:**

Extend the cost function with a group-aware intrusion 
term. When the tracker identifies a cohesive cluster of 
pedestrians $\mathcal{G} = \{p_1, \ldots, p_m\}$, 
compute its convex hull $\mathcal{H}(\mathcal{G})$ and 
add a group cost:

$$C^\mathcal{G}_t = \mu_g \cdot 
\frac{\max(0,\; -d(p_\text{ego},\, 
\mathcal{H}(\mathcal{G})))}{\text{Area}(\mathcal{H}
(\mathcal{G}))}$$

where $d(p_\text{ego}, \mathcal{H})$ is the signed 
distance to the hull boundary (negative inside, positive 
outside) and the area normalisation makes the penalty 
scale-invariant across group sizes.

The total cost becomes:

$$C_t = \mu \cdot d_{\text{intr},t} + 
\sum_{\mathcal{G}} C^\mathcal{G}_t$$

**Expected outcome:**

- Measurable reduction in group intrusion events (GI rate)
- More socially appropriate robot paths in group-dense 
environments
- No modification to the core RL architecture — purely 
a cost function extension
