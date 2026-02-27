# Probabilistic Task Offloading in D2D Edge Networks
**Ahmedabad University | CSE 400: Fundamentals of Probability in Computing**

---

## 📌 Project Context and Motivation
In the era of 5G and 6G, mobile applications such as Augmented Reality (AR), autonomous driving, and real-time speech recognition are becoming increasingly resource-intensive. Individual Mobile Devices (MDs) often lack the computational power to process these tasks within strict latency requirements. 

While offloading to the cloud is an option, it often introduces significant network latency. Our project explores **Device-to-Device (D2D) Edge Computing**, where idle peer devices in the vicinity act as service nodes. The critical challenge is the **Intrinsic Uncertainty** of the environment: wireless channels fade, task sizes vary, and peer devices have fluctuating workloads. Traditional deterministic models fail here; we address this by treating the entire system through a probabilistic lens.

---

## 🎯 Project Objective and Scope
The primary objective is to **Maximize the Global Task Completion Rate (TCR)**. 
Unlike standard systems that aim to minimize "Average Delay," our model recognizes that meeting a hard deadline ($\tau$) is a binary success/failure event. 

*   **Objective Function:** $\max_a \sum_{i \in N} E[Y_i]$
*   **Decision Problem:** Determining the optimal offloading target in a decentralized mesh.
*   **Metric:** Shifting the focus from "Mean Latency" to "Bernoulli Success Probability."

---

## 🎲 Sources of Uncertainty
To build a robust system, we identify and model the primary layers of randomness:
1.  **Processing Uncertainty (The Root Cause):** CPU time for a specific task (e.g., image recognition) varies based on input data complexity.
2.  **Traffic Uncertainty (Arrival Bursts):** User behavior is sporadic, meaning tasks do not arrive at fixed intervals.
3.  **Queueing Uncertainty:** Because arrivals and processing times are random, the waiting time in the buffer is highly volatile.
4.  **Channel Uncertainty:** Small-scale fading and shadowing in wireless links cause fluctuating transmission rates.
5.  **Device Heterogeneity:** Each peer device has a different computation capability ($c_i$), leading to different service rate distributions.

---

## 📊 Key Random Variables
To mathematically model the system, we define the following RVs:
*   **$X_{proc}$ (Task Processing Time):** A continuous RV representing service time.
*   **$T_{inter}$ (Inter-arrival Time):** Modeled as an Exponential distribution ($\sim Exp(\lambda)$).
*   **$W_q$ (Queue Waiting Time):** A derived RV representing buffer delay.
*   **$L_q$ (Queue Length):** Follows a Geometric distribution in our M/M/1 setup.
*   **$Y_i$ (Completion Indicator):** A Bernoulli RV ($1$ if Success, $0$ if Failure).

---

## 📐 Mathematical Modeling & Queueing Theory
Each mobile device is modeled as an **M/G/1 queueing system**. We utilize the Laplace-Stieltjes transform to derive the probability distribution of the sojourn time (total time).

### Completion Probability Formula
The probability that a task with deadline $\tau$ is completed on time is given by the Cumulative Distribution Function (CDF):
$$F(t \le \tau) = 1 - \exp(-(\mu - \lambda)\tau_{eff})$$

*   **$\mu$:** The service rate of the device.
*   **$\lambda$:** The total arrival rate (local + offloaded tasks).
*   **$\tau_{eff}$:** The effective time left ($\tau - t_{trans}$) after subtracting transmission delay.
*   **Stability Constraint:** The system is stable only if $\mu > \lambda$.

---

## 📉 Probabilistic Reasoning: Why Averages Lie
A cornerstone of our project is the application of **Jensen’s Inequality** to prove that designing for averages leads to system failure. 

In mobile edge networks, the "cost" of delay is a **convex function**—the penalty for being very late is disproportionately higher than the benefit of being early.
$$E[C(T)] \ge C(E[T])$$

**The Analogy:** If a teacher penalizes you $T^2$ points for being late, and you are 2 minutes late on average, you might expect a 4-point penalty. But if you are on time one day (0 points) and 4 minutes late the next (16 points), your average penalty is actually 8 points. **Variance increases cost.** Our implementation must simulate the full PDF variance to capture the true risk of failure.

---

## 🎮 Decision Logic: Game Theoretic Formulation
Since centralized control is impractical in a dynamic D2D mesh, we model the system as a **Non-Cooperative Strategic Game**.

### Nash Equilibrium
Devices act as rational players, reacting to the "Neighbor Load" to make an optimal local offloading decision. The system reaches a **Nash Equilibrium** when no single device can improve its task completion probability by unilaterally changing its strategy.

### Potential Function ($\Phi$)
To prove the system converges to stability, we use a **Potential Game** framework. We define a global potential function $\Phi(a)$ that captures aggregate network congestion. Whenever a device makes a decision that improves its own success rate, the global potential strictly increases:
$$\Phi(a_{new}) - \Phi(a_{old}) > 0 \implies \text{System Converges}$$

---

## 🚀 Planned Refinements
*   **Evolution to M/G/1:** Moving beyond Exponential simplifications to handle "heavy-tailed" risks (tasks like speech recognition that follow Gamma/Gaussian distributions).
*   **Solving "Stale Information":** Addressing the ambiguity that occurs when a device bases its offloading decision on outdated neighbor workload data.
*   **Dynamic Calibration:** Accounting for CPU throttling where processing capability ($\mu$) changes based on device thermal states.

---
**Base Paper:** [View Technical Reference](./base_paper/base_paper_Probabilistic_Task_Offloading_with_Uncertain_Processing_Times_in_Device-to-Device_Edge_Networks.pdf)  
*Last Updated: February 26, 2026*
