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

### 2.3 The "Variance Lesson": Jensen’s Inequality
A key theoretical contribution of our Milestone 2 report was the proof that **Averages Lie**. We utilized **Jensen's Inequality** to demonstrate that in a convex cost system (where delay cost is non-linear), designing for the average execution time leads to systemic failure:
$$\mathbb{E}[C(T)] \geq C(\mathbb{E}[T])$$
This proved that the **variance** of processing time is a greater risk factor than the **mean**.

### 2.4 Strategic Modeling: Potential Games
Because individual offloading decisions affect neighbor workloads, we identified a **Congestion Externality**. We modeled the interaction as a **Non-Cooperative Strategic Game** and utilized the concept of a **Potential Game** to prove that the system will always converge to a stable **Nash Equilibrium**.

---
## Milestone 3: Simulation, Computation & Empirical Validation

In this milestone, we operationalized our theoretical framework into a high-fidelity **Python-based Discrete-Event Simulator**. We built a "Scientific Laboratory" to test if a simple, deterministic rule could survive the unpredictable bursts of a real-world D2D mesh.

### 3.1 The Simulation Pipeline
To ensure statistical convergence and "Pro-Level" accuracy, we utilized the **Monte Carlo Method**:
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

### 3.4 Key Empirical Findings (The "Pro" Dashboard)
Our Stage 2 Analysis revealed critical system behaviors:
*   **The "Herd Effect":** Our purple Dashboard curve shows queue delays skyrocketing to **50+ seconds**. This proves that deterministic rules cause nodes to "synchronize" and overwhelm the same neighbor simultaneously, leading to system-wide saturation.
*   **Empirical Proof of the Jensen Gap:** By tracking the square of the delay ($T^2$), our red Dashboard curve proves that as variance increases, the "cost of being late" grows non-linearly. This confirms our M2 mathematical hypothesis.
*   **Baseline Failure:** The system's global **Task Completion Rate (TCR) collapsed** under high load, establishing a clear performance floor.

### 3.5 Conclusion & Next Steps
Milestone 3 proves that **Deterministic Logic is insufficient** for high-load D2D environments. This "Spectacular Failure" provides the scientific justification for **Milestone 4**, where we will discard myopic rules and introduce **Randomized AI Algorithms (DRL)** to break node synchronization.

---
