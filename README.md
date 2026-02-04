## Milestone 1 – Probabilistic Task Offloading in D2D Edge Networks

### Project Overview

This project studies task offloading in Device-to-Device (D2D) edge computing networks under uncertainty.
The main focus is on how uncertain task processing time and network conditions affect whether tasks can be completed before their deadlines. A probabilistic approach is used instead of fixed execution times.

### Objective

The objective of Milestone 1 is to understand the system from a probabilistic point of view and identify:

* The core problem in the system
* Sources of uncertainty
* Key random variables
* How probabilistic reasoning supports offloading decisions

### System Description

In the system, mobile devices generate computation-intensive tasks such as image processing or speech recognition.
Tasks can be executed locally or offloaded to nearby devices using D2D communication. Each task has a deadline, and system performance depends on whether the task finishes before this deadline.

### Sources of Uncertainty

The project considers multiple sources of uncertainty:

* Task processing time varies based on task type and input data
* Task arrivals are random over time
* Queueing and contention occur when multiple tasks compete for the same device
* Transmission delay changes with network conditions
* Devices have different computation capabilities

### Key Random Variables

The main random variables used in Milestone 1 are:

* Task Processing Time
* Task Arrival Time
* Task Completion Time
* Task Completion Indicator (completed before deadline or not)
* Device workload

### Probabilistic Modeling

* Task arrivals are modeled using a Poisson process
* Task processing time is modeled using probability distributions such as exponential, gamma, or Gaussian depending on application type
* An M/G/1 queueing model is used to relate arrival rate and processing time to completion probability
* Completion rate is computed as the probability that a task finishes before its deadline

### Decision Making

Each device estimates the probability of successful task completion for different offloading options.
Offloading decisions are made to maximize this probability.
A game-theoretic approach is used, where devices update decisions until a stable state (Nash equilibrium) is reached.

### Files Included

* **Concept Map**: Visual explanation of the system, uncertainty, random variables, probabilistic models, and decisions
* **Scribe Report**: Detailed written explanation of probabilistic reasoning and assumptions
* **Presentation (PPT)**: Summary of the system, models, and Milestone 1 understanding
* **Base Paper**: Reference research paper used for understanding and modeling

### Current Scope

This milestone focuses on high-level understanding and simplified assumptions.
More detailed modeling and refinements will be done in later milestones.
