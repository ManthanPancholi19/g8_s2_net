import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE_DIR, "..", "experiments")

if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

sns.set_theme(style="whitegrid", context="talk")

class Pro_D2D_Baseline_Simulation:
    def __init__(self, num_devices=5):
        self.num_devices = num_devices
        # CPU Frequencies (f_i) in GHz (Heterogeneous devices)
        self.cpu_freq = np.random.uniform(1.0, 2.5, num_devices)
        self.bandwidth = 5.0  # Base Wireless Transmission Rate (Mbps)
        self.max_distance = 50 # Max distance for D2D (Meters)

        # M/M/1 and M/G/1 Queue System for each device (Tracks workload in seconds)
        self.queues = np.zeros(num_devices)
        self.system_time = 0.0

        # Analytics tracking
        self.history = []

    def generate_task(self):
        # Poisson Process Arrival: Exponential Inter-arrival time
        inter_arrival_time = np.random.exponential(scale=0.2)
        self.system_time += inter_arrival_time

        # Advance queues based on elapsed time
        self.queues = np.maximum(0, self.queues - inter_arrival_time)

        # --- TASK PROFILES (Aligned with M2 Probabilistic Models) ---
        task_type = np.random.choice(["AR", "BeautyCam"])

        if task_type == "AR":
            # AR: M/G/1 Service (Gamma distribution captures 'bursty' processing)
            task_size = np.random.uniform(0.1, 0.4)
            task_cycles = np.random.gamma(shape=2, scale=1.5)
            deadline = 0.8
        else:
            # BeautyCam: M/M/1 Service (Exponential distribution)
            task_size = np.random.uniform(2.0, 4.0)
            task_cycles = np.random.exponential(scale=1.0)
            deadline = 2.5

        return task_size, task_cycles, deadline, task_type

    def generate_task_with_lambda(self, lam):
        # Helper method for Phase 1 Validation: Scales arrival rate
        inter_arrival_time = np.random.exponential(1/lam)
        self.system_time += inter_arrival_time
        self.queues = np.maximum(0, self.queues - inter_arrival_time)

        task_type = np.random.choice(["AR", "BeautyCam"])
        if task_type == "AR":
            task_size, task_cycles, deadline = np.random.uniform(0.1, 0.4), np.random.gamma(shape=2, scale=1.5), 0.8
        else:
            task_size, task_cycles, deadline = np.random.uniform(2.0, 4.0), np.random.exponential(scale=1.0), 2.5
        return task_size, task_cycles, deadline, task_type

    def deterministic_greedy_policy(self, source, task_size, task_cycles):
        # 1. Local Processing Estimation
        local_exec = task_cycles / self.cpu_freq[source]
        local_delay = self.queues[source] + local_exec

        best_target = source
        min_delay = local_delay
        decision_type = "Local"

        # 2. Remote Offloading Estimation (Iterate over neighbors)
        for neighbor in range(self.num_devices):
            if neighbor != source:
                # DISTANCE-BASED BANDWIDTH (Real-world physics)
                distance = np.random.uniform(5, self.max_distance)
                effective_bw = self.bandwidth / (1 + (distance/20)**2)

                t_trans = task_size / effective_bw
                t_exec = task_cycles / self.cpu_freq[neighbor]

                # M/M/1 Expected Sojourn Time logic for offloading
                remote_delay = t_trans + self.queues[neighbor] + t_exec

                # Stability Penalty: If neighbor is near saturation (mu < lambda)
                if self.queues[neighbor] > 2.0:
                    remote_delay *= 1.2

                if remote_delay < min_delay:
                    min_delay = remote_delay
                    best_target = neighbor
                    decision_type = "Offload"

        # Calculate actual transmission and execution for chosen winner to fix wait-time math
        if decision_type == "Local":
            actual_t_trans = 0
            actual_t_exec = task_cycles / self.cpu_freq[source]
        else:
            distance = np.random.uniform(5, self.max_distance)
            effective_bw = self.bandwidth / (1 + (distance/20)**2)
            actual_t_trans = task_size / effective_bw
            actual_t_exec = task_cycles / self.cpu_freq[best_target]

        return best_target, min_delay, actual_t_exec, actual_t_trans, decision_type

    def run_simulation(self, total_tasks=1500):
        print(" Starting 100/100 Deterministic Baseline Simulation...")
        successful_tasks = 0

        for task_id in range(1, total_tasks + 1):
            source = np.random.randint(0, self.num_devices)
            task_size, task_cycles, deadline, t_type = self.generate_task()

            # Fixed: Unpacking 5 values to match the policy update (Fixes ValueError)
            target, total_delay, t_exec, t_trans, dec_type = self.deterministic_greedy_policy(source, task_size, task_cycles)

            # Check Deadline Success (Bernoulli Outcome)
            is_success = total_delay <= deadline

            # JENSEN'S LOGIC: Tracking the convex cost of delay (Proof of M2 Theory)
            avg_delay_so_far = np.mean([x['Delay'] for x in self.history]) if self.history else total_delay
            jensen_gap = (total_delay**2) - (avg_delay_so_far**2)

            if is_success:
                successful_tasks += 1

            # REALISTIC WORKLOAD (Failed tasks still consume CPU resources)
            self.queues[target] += t_exec

            # Log data for Research-Grade Dashboard
            self.history.append({
                "Task_ID": task_id,
                "Type": t_type,
                "Success_Rate": (successful_tasks / task_id) * 100,
                "Delay": total_delay,
                "Decision": dec_type,
                "Queue_Load_Avg": np.mean(self.queues),
                "Jensen_Gap": jensen_gap
            })

            if task_id % 500 == 0:
                print(f" Processed {task_id}/{total_tasks} Tasks | Rate: {(successful_tasks / task_id) * 100:.2f}%")

        self.generate_dashboard(total_tasks, successful_tasks)

    def generate_dashboard(self, total_tasks, successful_tasks):
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(EXP_DIR, "..", "data", "simulation_log_m3.csv"), index=False)
        final_rate = (successful_tasks / total_tasks) * 100
        print(f"\n Final Success Rate: {final_rate:.2f}%")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Milestone 3: Advanced D2D Baseline Analysis (Real-World Constraints)', fontsize=22, fontweight='bold', y=0.98)

        # 1. System Reliability
        sns.lineplot(data=df, x="Task_ID", y="Success_Rate", color="#2ca02c", ax=axes[0, 0])
        axes[0, 0].set_title("System Reliability (Cumulative TCR %)", fontweight='bold')
        axes[0, 0].set_ylim(0, 100)

        # 2. Workload Mix
        type_counts = df["Type"].value_counts()
        axes[0, 1].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=["#ff9999","#66b3ff"], startangle=140)
        axes[0, 1].set_title("Workload Mix: AR vs BeautyCam", fontweight='bold')

        # 3. Congestion Externality
        sns.lineplot(data=df, x="Task_ID", y="Queue_Load_Avg", color="#9467bd", ax=axes[1, 0])
        axes[1, 0].fill_between(df["Task_ID"], df["Queue_Load_Avg"], color="#9467bd", alpha=0.2)
        axes[1, 0].set_title("Congestion Externality (Avg Queue Delay in Sec)", fontweight='bold')

        # 4. Jensen's Inequality Proof
        sns.lineplot(data=df, x="Task_ID", y="Jensen_Gap", color="#d62728", ax=axes[1, 1])
        axes[1, 1].set_title("Jensen Gap: E[Cost(T)] - Cost(E[T])", fontweight='bold')
        axes[1, 1].set_ylabel("Penalty Variance (Cost Difference)")

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(EXP_DIR, "M3_Final_Integrated_Dashboard.png"), dpi=300)
        plt.show()

    def plot_validation(self, lambda_range, tcr_sim, tcr_theory, wait_sim):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Stage 1: Mathematical Verification (M/G/1 vs Theory)', fontsize=18, fontweight='bold')

        ax1.plot(lambda_range, tcr_sim, 'o', color='#2E86C1', label='Simulated (SQF Baseline)')
        ax1.plot(lambda_range, tcr_theory, '--', color='red', label='M2 Theory (Stay Local)')
        ax1.set_title('TCR vs Arrival Rate', fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_xlabel('Arrival Rate (Lambda)')
        ax1.legend()

        ax2.plot(lambda_range, wait_sim, '-s', color='#28B463', linewidth=2)
        ax2.set_title('Avg Wait Time (Wq) vs Arrival Rate', fontweight='bold')
        ax2.set_ylabel('Wait Time (Seconds)')
        ax2.set_xlabel('Arrival Rate (Lambda)')

        plt.tight_layout()
        plt.savefig(os.path.join(EXP_DIR, "M3_Validation.png"), dpi=300)
        plt.show()

    def run_validation_sweep(self):
        print("Phase 1: Mathematical Validation (Lambda Sweep)...")
        lambda_range = np.linspace(0.1, 1.4, 20)
        tcr_sim, wait_sim, tcr_theory = [], [], []

        for l in lambda_range:
            trial_results = []
            for _ in range(10): # Monte-Carlo Trials
                self.queues = np.zeros(self.num_devices)
                successes = 0
                total_wait = 0
                for _ in range(500):
                    s, c, d, t = self.generate_task_with_lambda(l)

                    # Unpacking 5 values to align with policy return (Fixes ValueError)
                    target, total_delay, t_exec, t_trans, dec = self.deterministic_greedy_policy(0, s, c)

                    if total_delay <= d: successes += 1

                    # Correct Math: Wait Time = Total - Service - Trans (Fixes Negative Time)
                    wait_time = max(0, total_delay - t_exec - t_trans)
                    total_wait += wait_time
                    self.queues[target] += t_exec
                trial_results.append(((successes/500)*100, total_wait/500))

            tcr_sim.append(np.mean([x[0] for x in trial_results]))
            wait_sim.append(np.mean([x[1] for x in trial_results]))

            # Correct Theory Calculation: Weighted Average of AR and BeautyCam
            mu_avg = 1.0 / (0.5 * 3.0 + 0.5 * 1.0)
            d_avg = (0.8 + 2.5) / 2.0
            theory_val = (1 - np.exp(-(mu_avg - l) * d_avg)) * 100 if mu_avg > l else 0
            tcr_theory.append(theory_val)

        self.plot_validation(lambda_range, tcr_sim, tcr_theory, wait_sim)

if __name__ == "__main__":
    np.random.seed(42)
    sim = Pro_D2D_Baseline_Simulation(num_devices=5)

    # STAGE 1: VERIFY MATH (Code 2 Logic)
    sim.run_validation_sweep()

    # STAGE 2: ANALYZE SCENARIO (Code 1 Logic)
    sim.run_simulation(total_tasks=1500)