# Probabilistic Task Offloading in D2D Edge Networks
**Course:** CSE 400: Fundamentals of Probability in Computing  
**Group:** 8 (Networking)  
**Team:** Falak Shah, Ishani Prajapati, Nikunj Kalariya, Manthan Pancholi, Urvil Chodvadiya

---

## Project Vision & Scope
In high-density 5G/6G mobile networks, individual devices (MDs) are often overwhelmed by latency-sensitive applications like Augmented Reality (AR) and Speech Recognition. While Device-to-Device (D2D) offloading allows resource sharing, traditional systems fail because they treat execution time as a fixed constant.

This project replaces the "average delay" mindset with a **Probabilistic Framework**. We treat task processing, arrivals, and transmission as stochastic processes, aiming to maximize the global **Task Completion Rate (TCR)** rather than just speed.

---

## Milestone 1: Kickstart & Problem Formulation

### 1.1 The Core Probabilistic Problem
The fundamental challenge identified is that **Task Processing Time (TPT)** is a random variable ($T \sim f_T(t)$), not a constant. 
* In real-world scenarios, the same piece of code produces different runtimes based on input complexity. 
* **Example:** A Visual Relationship Detection (VRD) task processing an image with 1 target is 100x faster than an image with 20 targets.

### 1.2 The "Highway Toll Booth" Analogy
To illustrate the failure of deterministic logic, we modeled a highway toll booth system:
1. **The Assumption:** The system assumes every vehicle takes 10 seconds to pass.
2. **The Reality:** A heavy truck (complex task) unexpectedly takes 60 seconds.
3. **The Result:** This creates a "Traffic Jam" for all subsequent tasks in the queue, causing a chain reaction of deadline violations even for simple tasks. 

### 1.3 Identified Sources of Uncertainty
We mapped four primary layers of randomness that our system must manage:
* **Processing Uncertainty:** CPU time varies per task execution.
* **Queueing & Contention Delay:** Unpredictable waiting times in buffers.
* **Device Heterogeneity:** Varying CPU frequencies (1.6 - 4.8 GHz) across the mesh.
* **Transmission Delay:** Fluctuating bandwidth based on physical distance.

### 1.4 Formal System Objective
Instead of minimizing average latency, we formulated our objective as a **Bernoulli Success Problem**. Our goal is to maximize the expected number of tasks that finish before their strict deadline $\tau$:

$$\max_{\mathbf{a}} \sum_{i \in N} \mathbb{E}[Y_i]$$

Where:
- $Y_i = 1$ if $T_{sojourn,i} \leq \tau_i$ (Success)
- $Y_i = 0$ otherwise (Failure)

---
##  Milestone 2: Mathematical Modeling & Formulation

In this stage, we transitioned from abstract concepts to a concrete mathematical framework using **Queueing Theory** and **Stochastic Calculus**. We formally defined the system dynamics and derived the success probability equations.

### 2.1 Formal Definition of Random Variables
To strictly model the system, we defined the following continuous and discrete RVs:
*   **$X_{proc}$ (Task Processing Time):** Modeled as an Exponential distribution ($Exp(\mu)$) based on device service rate.
*   **$T_{inter}$ (Inter-arrival Time):** Modeled as $Exp(\lambda)$, establishing a Poisson arrival process.
*   **$W_{q}$ (Queue Waiting Time):** A derived RV representing the buffer delay, vital for capturing system congestion.
*   **$T_{sojourn}$ (Total Delay):** The aggregate random variable defined as $W_q + X_{proc} + t_{trans}$.
*   **$Y_i$ (Success Indicator):** A Bernoulli random variable where $Y_i \sim \text{Bernoulli}(P_i)$.

### 2.2 Mathematical Realization: The M/M/1 Queue
We modeled each edge device as an **M/M/1 queueing system** (Markovian arrivals, Markovian service, 1 server). 
*   **Sojourn Time Density:** For a stable system ($\mu > \lambda$), the probability density of the sojourn time is:
    $$f_T(t) = (\mu - \lambda)e^{-(\mu - \lambda)t}$$
*   **Closed-Form Completion Rate:** By integrating the density, we derived the formula used in our simulation for a task with deadline $\tau$:
    $$F(t \leq \tau) = 1 - e^{-(\mu - \lambda)\tau_{eff}}$$
    *Where $\tau_{eff} = \tau - t_{trans}$ (Effective time remaining after transmission).*

### 2.3 The "Variance Lesson": Jensen's Inequality
A key theoretical contribution of our Milestone 2 report was the proof that **Averages Lie**. We utilized **Jensen's Inequality** to demonstrate that in a convex cost system (where delay cost is non-linear), designing for the average execution time leads to systemic failure:
$$\mathbb{E}[C(T)] \geq C(\mathbb{E}[T])$$
This proved that the **variance** of processing time is a greater risk factor than the **mean**.

### 2.4 Strategic Modeling: Potential Games
Because individual offloading decisions affect neighbor workloads, we identified a **Congestion Externality**. We modeled the interaction as a **Non-Cooperative Strategic Game** and utilized the concept of a **Potential Game** to prove that the system will always converge to a stable **Nash Equilibrium**.

---
## Milestone 3: Simulation, Computation & Empirical Validation

In this milestone, we operationalized our theoretical framework into a high-fidelity **Python-based Discrete-Event Simulator**. We built a "Scientific Laboratory" to test if a simple, deterministic rule could survive the unpredictable bursts of a real-world D2D mesh.

### 3.1 The Simulation Pipeline
To ensure statistical convergence and  accuracy, we utilized the **Monte Carlo Method**:
*   **Scale:** Simulated **600,000 individual tasks** (Sweep of 20 Arrival Rates $\times$ 10 independent trials $\times$ 3,000 tasks per run).
*   **Architecture:** The code tracks the `device_free_time` for each heterogeneous node, accurately calculating wait times ($W_q$) and sojourn times ($T_{sojourn}$).

### 3.2 Deterministic Baseline: Shortest Delay First (SDF)
Following the M3 guidelines, we implemented a non-randomized, myopic greedy rule. When a task arrives at time $t$, the system performs a **State-Space Assessment**:
1.  **Calculates Expected Delay:** $E[D] = W_q + t_{trans} + t_{exec}$.
2.  **Applies Stability Constraint:** A **1.2x Penalty** is added if a neighbor's queue backlog exceeds 2.0s to simulate saturation risk.
3.  **The Decision:** The task is routed to the device offering the absolute minimum estimated delay.

### 3.3 High-Fidelity Stochastic Profiling
We moved beyond simple averages by implementing a **Bimodal Workload Profile** to simulate application diversity:
*   **AR Tasks (M/G/1):** Modeled via **Gamma Distribution** ($\alpha=2, \beta=1.5$) to capture "bursty" computation.
*   **BeautyCam (M/M/1):** Modeled via **Exponential Distribution** ($\mu=1.0$) for standard Markovian service.
*   **Physical Layer:** Distance is modeled as $d \sim U(5, 50)$, which dynamically degrades effective bandwidth and increases transmission uncertainty.

### 3.4 Key Empirical Findings 
Our Stage 2 Analysis revealed critical system behaviors:
*   **The "Herd Effect":** Our purple Dashboard curve shows queue delays skyrocketing to **50+ seconds**. This proves that deterministic rules cause nodes to "synchronize" and overwhelm the same neighbor simultaneously, leading to system-wide saturation.
*   **Empirical Proof of the Jensen Gap:** By tracking the square of the delay ($T^2$), our red Dashboard curve proves that as variance increases, the "cost of being late" grows non-linearly. This confirms our M2 mathematical hypothesis.
*   **Baseline Failure:** The system's global **Task Completion Rate (TCR) collapsed** under high load, establishing a clear performance floor.

### 3.5 Conclusion & Next Steps
Milestone 3 proves that **Deterministic Logic is insufficient** for high-load D2D environments. This "Spectacular Failure" provides the scientific justification for **Milestone 4**, where we will discard myopic rules and introduce **Randomized AI Algorithms (DRL)** to break node synchronization.

---

## Milestone 4: Randomized Algorithm Strategy & Multi-Metric Performance Evaluation

Milestone 4 addresses the "Herd Effect" failure identified in Milestone 3 by introducing **principled randomness** into the offloading decision. The deterministic SDF rule is replaced with a Double Deep Q-Network (DDQN) that samples actions from a learned probability distribution, breaking node synchronization and restoring system stability.

### 4.1 Why Randomization?

The Herd Effect is a **coordination failure, not an estimation problem**. Every deterministic agent applying the same greedy rule converges on the same target simultaneously, creating a negative probabilistic dependency:

$$P(\text{Succ}_A \cap \text{Succ}_B) < P(\text{Succ}_A) \cdot P(\text{Succ}_B)$$

Introducing stochasticity breaks this synchronization. Even agents sharing identical state observations draw different samples from the Boltzmann distribution, automatically distributing burst load across the network.

### 4.2 The Transition to Stochastic Policy

*   **M3 Deterministic:** $Action = \arg\min_j D_j$ — fixed function of state
*   **M4 Randomized:** $Action \sim \pi(A \mid S)$ — random variable drawn from a learned distribution

### 4.3 Algorithm Design: DDQN with Epsilon-Boltzmann Exploration

*   **Exploration Strategy:** An Epsilon-Boltzmann hybrid. With probability $\epsilon_t$ the agent explores via Boltzmann sampling; otherwise it exploits the learned greedy action.
*   **Boltzmann Sampling:**
    $$P(A = j \mid S = s) = \frac{\exp(Q_\theta(s, j) / T_{temp})}{\sum_{k=1}^{N} \exp(Q_\theta(s, k) / T_{temp})}, \quad T_{temp} = 0.5$$
*   **Epsilon Decay:** $\epsilon_t = \max(0.1,\ 0.98^t)$, decaying from 1.0 to 0.1 over 50 episodes
*   **Neural Architecture:** $\mathbb{R}^8 \rightarrow 128 \rightarrow 128 \rightarrow \mathbb{R}^5$ (ReLU layers, one Q-value per target node)
*   **State Vector:** $S_t = [W_{q,1}, \ldots, W_{q,5},\ \text{src},\ X,\ Y,\ \tau] \in \mathbb{R}^8$
*   **Double Q-Learning Update:**
    $$L(\theta) = \mathbb{E}\left[\left(r + \gamma \cdot Q_{\theta^-}\!\left(s', \arg\max_{a'} Q_\theta(s', a')\right) - Q_\theta(s, a)\right)^2\right]$$
*   **Hyperparameters:** $\gamma = 0.95$, lr $= 0.001$ (Adam), batch $= 64$, replay buffer $= 5{,}000$ transitions

### 4.4 Probabilistic Modeling: Correlated Input Structure

Milestone 4 explicitly models the **non-factorizable joint distribution** of task features:

$$Y = 5X + \epsilon, \quad \epsilon \sim \mathcal{N}(0.5, 1.95), \quad \rho(X, Y) \approx +0.55$$
$$f(X, Y) \neq f(X) \cdot f(Y)$$

Large tasks (high $X$) simultaneously carry above-average CPU demand (high $Y$), producing doubly intense bursts. The DDQN state encodes both $X$ and $Y$, allowing the learned policy to account for this coupled structure.

### 4.5 The Reward Function & Jensen's Inequality

$$r(s, a) = \begin{cases} 50 / \exp(W_q) & \text{if } T_{soj} \leq \tau \\ -(T_{soj}^2 + \exp(W_q)) & \text{if } T_{soj} > \tau \end{cases}$$

The exponential penalty $\exp(W_q)$ mirrors the congestion externality. The agent learns to seek under-utilized nodes even when they are not the globally shortest-delay option — directly addressing the failure mode Jensen's Inequality predicted.

### 4.6 Evaluation Metrics

To evaluate whether the randomized strategy performs better than the deterministic baseline, the following metrics were defined:

| Metric | Definition | Why Appropriate |
| :--- | :--- | :--- |
| **Task Completion Rate (TCR)** | $\Phi = \mathbb{E}[\sum_i Y_i]$ | Primary objective — directly measures deadline compliance |
| **Tail Latency (95th pct.)** | $t : P(T_{soj} > t) = 0.05$ | Averages mislead; tail behavior reveals true policy quality per Jensen's Inequality |
| **Jain's Fairness Index** | $J = (\sum_j c_j)^2 / (N \sum_j c_j^2)$ | Detects load imbalance caused by the Herd Effect; $J=1$ is perfectly equitable |
| **System Throughput** | $\Theta = \sum_i Y_i / T_{system}$ | Confirms sustained performance under high arrival rate $\lambda$ |
| **Queue Stability ($W_q$)** | Mean backlog across all nodes | Directly measures whether synchronization collapse is broken |
| **Strategy Mix** | Fraction of explore vs. exploit decisions | Confirms the stochastic policy is operational — neither fully random nor fully greedy |

### 4.7 How to Run

```bash
cd code/src/milestone4
python m4_simulation.py
```

Automatically saves:
- `code/experiments/M4_Final_Dashboard.png` — 8-panel evaluation dashboard
- `code/data/m4_episode_results.csv` — per-task metrics from the final episode
- `code/data/m4_xy_pairs.csv` — correlated (X, Y) task pairs, $\rho \approx +0.55$

### 4.8 Comparative Results: SDF vs. DDQN

| Metric | M3 Deterministic (SDF) | M4 Randomized (DDQN) |
| :--- | :--- | :--- |
| **Task Completion Rate (TCR)** | < 1% at saturation | > 80% post-training |
| **Tail Latency (95th pct.)** | > 50.0 seconds | < 6.0 seconds |
| **Jain's Fairness Index** | $\approx 0.2$ (Severe imbalance) | $\approx 0.6$ (Balanced load) |
| **System Throughput** | Collapses at high $\lambda$ | Stable $\approx 0.30$ tasks/sec |
| **Queue Stability ($W_q$)** | Divergent ($W_q > 50s$) | Convergent ($W_q < 1.5s$) |
| **Strategy Mix** | 100% Exploit | 12.6% Explore / 87.4% Exploit |

> **Note:** The sole architectural change between M3 and M4 is SDF replaced by DDQN Epsilon-Boltzmann. All other simulation parameters are identical. All improvements are causally attributable to the randomized policy alone.

### 4.9 Key Insights from Randomized Strategy
1.  **Randomness as Coordination:** Purposeful load diversification, achieved through an exploration fraction of ~12.6%, is sufficient to break the synchronization that caused M3 to collapse.
2.  **Experience Replay is Essential:** Uniform sampling from the 5,000-transition replay buffer breaks the temporal autocorrelation in queue backlogs, enabling stable convergent learning.
3.  **Hedging Against Tail Events:** By dropping 95th-percentile latency from 50s to < 6s, the DDQN validates the theoretical need to optimize for variance rather than just means (Jensen's Inequality).

---

## Milestone 5: Future Refinements & Remaining Challenges

While Milestone 4 proves the efficacy of randomized strategies, it identifies several constraints to be addressed in the final project phase:

1.  **POMDP Extension:** Transition from oracle queue access to a Partially Observable Markov Decision Process (POMDP) where state broadcasts arrive with delay $\delta_{comm}$.
2.  **Non-Stationary Traffic:** Evaluating the policy under bursty, time-varying arrival rates ($\lambda(t)$) that may surge $5\times$ during peak hours.
3.  **Network Scaling:** Scaling the five-node network to $N=20$ nodes, requiring Attention Mechanisms or Graph Neural Networks (GNNs) to handle the expanded state space.
4.  **Stochastic Channel Model:** Incorporating Rayleigh fading into the transmission delay $D_{tx}$ to introduce a genuine source of sojourn variance at the physical layer.
