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
in deploying mobile robots in human-populated environments. 
Treating humans as static obstacles fails because human motion 
is intentional, social, and group-structured.

This project addresses that challenge through a system that 
combines:

- **Multi-agent trajectory prediction** — forecasting future 
pedestrian positions using attention-based graph networks 
that model both pairwise and group-wise interactions
- **Uncertainty quantification** — wrapping predictions with 
statistically valid uncertainty bounds via Adaptive Conformal 
Inference (ACI), so the robot knows *how much to trust* each 
prediction at every timestep
- **Constrained reinforcement learning** — a policy that 
explicitly accounts for prediction uncertainty in its cost 
function, learning to stay safe even when predictions are 
wrong or crowd dynamics shift

The system generalizes across out-of-distribution scenarios — 
faster pedestrians, different crowd behavior models, pedestrian 
group dynamics — where standard RL navigation methods degrade 
significantly. The learned policy was deployed on a physical 
Mecanum-wheel robot in real outdoor crowd environments without 
any fine-tuning, demonstrating direct sim-to-real transfer.

---

## Core System — Designed and Built by the Lab

> The following components form the core of the project. 
They were designed and implemented by PhD researchers at TASL. 
They are documented here to provide complete project context. 
My specific contributions are described in the 
[My Contributions](#my-contributions) section.

### Trajectory Prediction

Two predictors are used interchangeably to show 
model-agnosticism:

- **Constant Velocity (CV)** — rule-based linear extrapolation 
from current velocity; fast and interpretable
- **Gumbel Social Transformer (GST)** — learning-based, 
handles sparse interactions and partially-detected agents 
in dense crowds

Both output $K$-step future positions $\hat{p}_{h,k}(t)$ 
for each pedestrian $h$.

### Adaptive Conformal Inference (ACI)

Raw predictions are wrapped with uncertainty bounds using 
**DtACI** — an online, distribution-free method that adapts 
as crowd dynamics shift.

Actual prediction error at each step:

$$\delta_{h,k}(t) = \| p_h(t) - \hat{p}_{h,k}(t-k) \|_2$$

Estimated uncertainty updated across $M$ parallel estimators:

$$\hat{\delta}^{(m)}_{h,k}(t) = \hat{\delta}^{(m)}_{h,k}(t-1) 
- \gamma^{(m)} \bigl(\alpha - \text{err}^{(m)}_{h,k}(t)\bigr)$$

Best estimator selected via exponential weighting using the 
pinball loss:

$$\ell(\delta, \hat{\delta}) = \begin{cases} 
\alpha(\delta - \hat{\delta}) & \delta \geq \hat{\delta} \\ 
(\alpha-1)(\delta - \hat{\delta}) & \delta < \hat{\delta}
\end{cases}$$

Coverage parameter $\alpha = 0.1$ gives 90% coverage. 
The resulting uncertainty radius $\hat{\delta}_{h,k}$ 
expands automatically when crowd dynamics shift.

### Safety-Critical Area Design

Each pedestrian's safety zone is the union of two sub-areas:

$$D_1(p_\text{ego}) = \{ p_\text{ego} : \|p_\text{ego} - 
p_h\| \leq r_\text{ego} + r_h + r_\text{comfort} \}$$

$$D_2(p_\text{ego}) = \{ p_\text{ego} : \|p_\text{ego} - 
\hat{p}_{h,k}\| \leq r_\text{ego} + r_h + \hat{\delta}_{h,k} 
\}$$

$D_2$ grows dynamically with prediction uncertainty — 
giving the robot a larger margin precisely when predictions 
are less reliable.

### Constrained Reinforcement Learning (CRL)

Navigation is formulated as a **Constrained MDP (CMDP)**. 
The original state space:

$$S_t = [\mathbf{e},\; \mathbf{h},\; \mathbf{m}]$$

where $\mathbf{e}$ is ego robot info, $\mathbf{h}$ is 
human physical state, and $\mathbf{m}$ contains 
model-generated features (trajectory predictions + 
ACI uncertainty estimates).

Optimization objective:

$$\max_\pi \; \mathbb{E}_\pi \Bigl[\sum_{t=0}^{T} R_t\Bigr] 
\quad \text{s.t.} \quad 
\mathbb{E}_\pi \Bigl[\sum_{t=0}^{T} d_{\text{intr},t}\Bigr] 
\leq \tilde{d}$$

Cost per timestep:

$$C_t(S_t, A_t) = \mu \cdot d_{\text{intr},t}$$

**PPO-Lagrangian** maintains a multiplier $\lambda$ to 
balance reward and cost. Combined advantage for the actor:

$$\hat{A}'_t = \frac{\hat{A}^R_t - \lambda\hat{A}^C_t}
{1 + \lambda}$$

$\lambda$ updated via:

$$\ell^\lambda_t = -\lambda\bigl(\bar{C} - \tilde{d}_C\bigr)$$

### Policy Network

Processes uncertainty-augmented observations through:

1. **H-H Attention** — multi-head attention over all 
pedestrian pairs modeling human-human interactions
2. **H-R Attention** — fuses robot state into 
human-centric embeddings
3. **GRU** — captures temporal dynamics across timesteps
4. **Actor + Reward Critic** (shared backbone) + 
**Cost Critic** (separate network)

---

## My Contributions

My work focused on extending the system's **perception 
infrastructure and environment mapping** — the input layer 
the RL policy operates on top of. Both contributions 
involve classical robotics components and directly interface 
with the robot learning system by modifying the RL agent's 
observation space.

---

### 1. 3D LiDAR + RGB-D Sensor Fusion

**Context:**
The published system used only a 2D LiDAR (RPLIDAR-A1) 
with DR-SPAAM — a learning-based human detector for 2D 
range data. A single horizontal scan plane limits 
robustness: pedestrians at varying heights, partially 
occluded agents, or those outside the scan plane are 
missed entirely. Multi-modal sensor fusion was explicitly 
flagged as future work in the original research.

**Approach — Hybrid Classical + Learning Pipeline:**

The extended perception pipeline fuses two modalities:

- **3D LiDAR** — provides full volumetric point cloud 
$\mathcal{P}_t = \{(x_i, y_i, z_i)\}$ capturing 
pedestrian geometry across multiple elevation planes
- **RGB-D Camera** — provides pixel-aligned depth map 
$D_t$ enabling detection in visually rich but 
geometrically sparse regions

The fusion pipeline:
```text
3D LiDAR Point Cloud ──► Euclidean Clustering / DBSCAN
│
RGB-D Depth Map ──► Depth Projection onto scan plane
│
┌────────────────────┘
▼
DR-SPAAM (Learning-Based Detector)
operating on projected multi-modal input
│
▼
SORT Tracker → Pedestrian State Estimates x_h(t)
│
▼
GST Predictor + DtACI → p̂_{h,k}(t), δ̂_{h,k}
```

---

Each pedestrian state estimate from the fused pipeline:

$$\mathbf{x}_h(t) = [p_x,\; p_y,\; v_x,\; v_y,\; 
c_h]^\top$$

where $c_h \in [0,1]$ is a detection confidence score 
derived from the fusion agreement between the LiDAR 
cluster and the depth image detection.

**Robot Learning Interface:**

This is where the contribution connects to the RL system. 
The fused pedestrian estimates $\mathbf{x}_h(t)$ replace 
the 2D LiDAR-only estimates in the CMDP state:

$$\mathbf{h}_\text{fused}(t) = 
\bigl[\mathbf{x}_1(t),\; \mathbf{x}_2(t),\; 
\ldots,\; \mathbf{x}_H(t)\bigr]$$

The RL policy's human observation component $\mathbf{h}$ 
in $S_t = [\mathbf{e}, \mathbf{h}, \mathbf{m}]$ is 
replaced by $\mathbf{h}_\text{fused}$ — richer, more 
geometrically complete pedestrian state estimates that 
directly improve what the policy reasons over. Extending 
the observation space of an RL agent is itself a robot 
learning design decision, as it changes what the policy 
can learn to attend to.

**What I contributed:**
- Contributed to integrating 3D LiDAR point cloud 
processing with the existing DR-SPAAM + SORT pipeline
- Supported the depth image fusion component feeding 
into the multi-modal detection front-end
- Worked on validating detection robustness improvements 
in partial occlusion and varied-height scenarios through 
simulation testing

---

### 2. SLAM-Based Environment Mapping — Hybrid Navigation

**Context:**
The original system modeled static obstacles only as 
circular agents — a simplification that fails in real 
environments with walls, corridors, and irregular 
structures. SLAM integration was explicitly flagged as 
future work in the original research.

**Approach — Classical SLAM + RL Hybrid:**

Navigation in this extended system operates across 
two layers:

- **Mapping layer (classical)** — Cartographer SLAM 
builds and maintains a globally consistent 2D occupancy 
grid $\mathcal{M}_t$ via graph-optimization on LiDAR 
scan data with loop closure
- **Decision layer (RL)** — the CRL policy continues 
to handle all dynamic obstacle avoidance and 
crowd navigation

The local map patch $\mathcal{M}_\text{local}(t)$ is 
extracted around the robot's current position as a 
fixed-size occupancy window:

$$\mathcal{M}_\text{local}(t) \in \{0,1\}^{W \times W}$$

where $W$ is the window width in grid cells and cell 
values indicate free (0) or occupied (1).

**CMDP State Extension:**

The SLAM-derived map is incorporated as an additional 
observation channel in the RL agent's state:

$$S_t^\text{extended} = [\mathbf{e},\; \mathbf{h},\; 
\mathbf{m},\; \mathcal{M}_\text{local}(t)]$$

This is a robot learning contribution — modifying the 
CMDP state representation changes what the policy can 
condition its decisions on. The policy can now distinguish 
between open space and wall-bounded corridors rather than 
treating all unoccupied space as equivalent.

**Extended Cost Function:**

An additional map-based cost term discourages proximity 
to static obstacles in the SLAM map:

$$C_t^\text{map} = \nu \cdot \max\bigl(0,\; 
d_\text{safe} - d_\text{obs}(p_\text{ego}(t), 
\mathcal{M}_t)\bigr)$$

where $d_\text{obs}$ is the distance to the nearest 
occupied cell in $\mathcal{M}_t$ and $d_\text{safe}$ 
is a minimum clearance threshold. The total cost becomes:

$$C_t^\text{total} = \mu \cdot d_{\text{intr},t} + 
C_t^\text{map}$$

This keeps the CRL constraint formulation intact while 
extending it to handle arbitrary obstacle geometry — 
purely a cost function modification with no changes 
to the underlying RL architecture.

**What I contributed:**
- Contributed to integrating Cartographer SLAM for 
live occupancy map generation in the navigation environment
- Supported modification of the CMDP observation space 
to incorporate the local map patch $\mathcal{M}_\text{local}$
- Assisted in extending the CRL cost function with the 
map-based obstacle term
- Validated the system in structured simulation environments 
with non-circular static obstacles including walls and 
corridor geometry

---

### 3. System Validation & Deployment Support

- Contributed to end-to-end system validation through 
simulation runs and real-world experiments on the 
ROSMASTER X3 platform
- Assisted in evaluating navigation safety and robustness 
metrics — success rate, collision rate, intrusion time 
ratio — across multiple crowd scenarios

---

## Hardware Stack

| Component | Details |
|-----------|---------|
| Robot Platform | ROSMASTER X3 — Mecanum-wheel mobile robot |
| Onboard Compute | NVIDIA RTX 3070 (Mobile) — connected via router |
| 2D LiDAR (baseline) | RPLIDAR-A1 — 360°, ~6 Hz scan rate |
| 3D LiDAR (extended) | 3D LiDAR — volumetric point cloud perception |
| RGB-D Camera (extended) | RGB-D Camera — pixel-aligned depth + colour |
| Locomotion | 4× Mecanum wheels — holonomic (independent vx, vy) |

---

## Software Stack

| Category | Tools |
|----------|-------|
| Middleware | ROS 2 |
| Simulation | CrowdNav (RL training), Gazebo |
| Deep Learning | PyTorch |
| RL Framework | PPO-Lagrangian via OmniSafe |
| Trajectory Prediction | Gumbel Social Transformer (GST) |
| Uncertainty Quantification | DtACI |
| Human Detection | DR-SPAAM (learning-based, 2D/3D range data) |
| Multi-Object Tracking | SORT |
| SLAM | Google Cartographer (ROS 2) |
| Languages | Python, C++ |

---

## Planned Improvements

These improvements are planned for my independent 
simulation reimplementation in ROS 2 + Gazebo.

---

### Improvement 1: Density-Adaptive Uncertainty Coverage

**Problem:**
The ACI module uses a fixed coverage parameter 
$\alpha = 0.1$ regardless of local crowd density. 
In sparse environments this is unnecessarily conservative. 
In highly dense crowds it may still be insufficient.

**Planned modification:**

Define a real-time local pedestrian density estimate:

$$\rho(t) = \left| \{ h : \|p_h(t) - p_\text{ego}(t)\| 
\leq r_\rho \} \right|$$

Drive $\alpha$ dynamically based on $\rho(t)$:

$$\alpha(t) = \alpha_{\min} + (\alpha_{\max} - 
\alpha_{\min}) \cdot \exp\!\bigl(-\beta\,\rho(t)\bigr)$$

Higher density $\rightarrow$ lower $\alpha$ $\rightarrow$ 
tighter uncertainty bounds $\rightarrow$ more conservative 
safety margins. Lower density $\rightarrow$ higher $\alpha$ 
$\rightarrow$ relaxed bounds $\rightarrow$ more efficient 
navigation.

The ACI update rule remains unchanged — only the coverage 
parameter that drives the quantile target becomes 
state-dependent.

**Expected outcome:**
- More natural navigation pacing across varied crowd densities
- Maintained safety guarantees in dense scenarios
- Reduced unnecessary slow-downs in sparse scenarios

---

### Improvement 2: Group-Density Aware Cost Shaping

**Problem:**
The CRL cost function penalizes intrusion into individual 
pedestrian uncertainty areas — but treats navigating 
near a single person identically to entering a cohesive 
group. Entering a group is more socially disruptive, 
harder to recover from, and can trap the robot.

**Planned modification:**

For each detected cohesive group 
$\mathcal{G} = \{p_1, \ldots, p_m\}$, compute its 
convex hull $\mathcal{H}(\mathcal{G})$ and add a 
group-intrusion cost:

$$C_t^{\mathcal{G}} = \mu_g \cdot 
\frac{\max\!\bigl(0,\; -d(p_\text{ego},\; 
\mathcal{H}(\mathcal{G}))\bigr)}
{\text{Area}\bigl(\mathcal{H}(\mathcal{G})\bigr)}$$

where $d(p_\text{ego}, \mathcal{H})$ is the signed 
distance to the hull boundary — negative inside, 
positive outside. Area normalization makes the penalty 
scale-invariant across group sizes.

Total extended cost:

$$C_t^\text{total} = \mu \cdot d_{\text{intr},t} + 
\sum_{\mathcal{G}} C_t^{\mathcal{G}}$$

This is a pure cost function extension — no changes 
to the underlying RL architecture required.

**Expected outcome:**
- Measurable reduction in group intrusion rate (GI metric)
- More socially appropriate paths in group-dense environments
- No degradation in individual collision avoidance metrics
