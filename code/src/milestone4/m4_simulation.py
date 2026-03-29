import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from scipy import stats

sns.set_theme(style="whitegrid", context="talk")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDQN_Agent:
    def __init__(self, state_size, action_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=5000)
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.1
        self.epsilon_decay = 0.98
        self.policy_net    = DQN(state_size, action_size)
        self.target_net    = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer     = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion     = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        if np.random.rand() < self.epsilon:
            probs  = torch.softmax(q_values / 0.5, dim=1).numpy()[0]
            action = np.random.choice(self.action_size, p=probs)
            return action, "Explore (Randomized)"
        else:
            return torch.argmax(q_values[0]).item(), "Exploit (AI)"

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states           = torch.FloatTensor(np.array(states))
        actions          = torch.LongTensor(actions).unsqueeze(1)
        rewards          = torch.FloatTensor(rewards).unsqueeze(1)
        next_states      = torch.FloatTensor(np.array(next_states))
        dones            = torch.FloatTensor(dones).unsqueeze(1)
        next_actions     = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values    = self.target_net(next_states).gather(1, next_actions)
        target_q_values  = rewards + (self.gamma * next_q_values * (1 - dones))
        current_q_values = self.policy_net(states).gather(1, actions)
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class M4_Simulation:
    def __init__(self, num_devices=5):
        self.num_devices = num_devices
        self.cpu_freq    = np.random.uniform(1.0, 2.5, num_devices)
        self.queues      = np.zeros(num_devices)
        self.system_time = 0.0
        self.history     = []
        self.agent       = DDQN_Agent(num_devices + 4, num_devices)
        self.xy_pairs    = []

    def generate_task(self):
        inter_arrival     = np.random.exponential(scale=1.5)
        self.system_time += inter_arrival
        self.queues       = np.maximum(0, self.queues - inter_arrival)
        task_size         = np.random.uniform(0.1, 1.0)
        task_cycles       = np.maximum(0.1, (task_size * 5) + np.random.normal(0.5, 1.95))
        deadline          = 3.0 if task_size < 0.5 else 5.0
        return task_size, task_cycles, deadline

    def execute(self, source, target, size, cycles, deadline):
        t_trans      = 0 if target == source else (size / 1.5)
        t_exec       = cycles / self.cpu_freq[target]
        total_delay  = t_trans + self.queues[target] + t_exec
        is_success   = total_delay <= deadline
        load_penalty = np.exp(self.queues[target])
        if is_success:
            reward = 50.0 / load_penalty
        else:
            reward = -1.0 * (total_delay ** 2 + load_penalty)
        return total_delay, t_exec, is_success, reward

    def train(self):
        print("=" * 60)
        print("  M4 DDQN Training — D2D Edge Network")
        print("=" * 60)

        for e in range(50):
            self.queues      = np.zeros(self.num_devices)
            self.system_time = 0.0
            successful       = 0
            delays, targets  = [], []

            if e == 49:
                self.agent.epsilon = 0.15
                self.xy_pairs      = []

            for t_id in range(500):
                size, cycles, dl = self.generate_task()
                source           = np.random.randint(0, self.num_devices)
                state            = np.array(list(self.queues) + [source, size, cycles, dl])
                target, d_type   = self.agent.act(state)
                delay, exec_t, success, reward = self.execute(source, target, size, cycles, dl)

                if success:
                    successful += 1
                self.queues[target] += exec_t
                delays.append(delay)
                targets.append(target)

                if e == 49:
                    self.xy_pairs.append((size, cycles))

                n_s, n_c, n_dl = self.generate_task()
                next_state = np.array(list(self.queues) + [source, n_s, n_c, n_dl])
                self.agent.remember(state, target, reward, next_state, t_id == 499)
                self.agent.replay(64)

                if e == 49:
                    counts   = np.bincount(targets, minlength=self.num_devices)
                    fairness = (np.sum(counts) ** 2) / (
                        self.num_devices * (np.sum(counts ** 2) + 1e-9)
                    )
                    self.history.append({
                        "Task_ID":      t_id,
                        "Success_Rate": (successful / (t_id + 1)) * 100,
                        "Latency":      np.mean(self.queues),
                        "Decision":     d_type,
                        "Tail":         np.percentile(delays, 95),
                        "Fairness":     fairness,
                        "Utilization":  np.mean(self.queues > 0.1),
                        "Throughput":   successful / max(self.system_time, 1e-9),
                    })

            if (e + 1) % 5 == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            print(f"  Episode {e+1:2d}/50 | TCR: {(successful/500)*100:6.2f}% | ε: {self.agent.epsilon:.4f}")

        self.plot()

    def plot(self):
        df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(2, 4, figsize=(28, 14))
        fig.suptitle(
            "Milestone 4: Randomized Algorithm Strategy — Multi-Metric Performance Evaluation\n"
            "Group 8 | CSE400 | Ahmedabad University",
            fontsize=28, fontweight="bold", y=1.02,
        )

        axes[0, 0].plot(df["Task_ID"], df["Success_Rate"], color="green", linewidth=3, label="M4 DDQN")
        axes[0, 0].axhline(y=0.40, color="red", linestyle="--", label="M3 SDF Baseline (~0.40%)")
        axes[0, 0].set_title("Reliability (TCR %)", fontweight="bold")
        axes[0, 0].set_xlabel("Task ID")
        axes[0, 0].set_ylabel("TCR (%)")
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].set_ylim(0, 105)

        axes[0, 1].plot(df["Task_ID"], df["Tail"], color="#d62728", linewidth=2)
        axes[0, 1].axhline(y=6.0, color="orange", linestyle="--", label="Target <6s", alpha=0.8)
        axes[0, 1].set_title("Worst-Case Tail Latency (95th pct)", fontweight="bold")
        axes[0, 1].set_xlabel("Task ID")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].legend(fontsize=10)

        df["Decision"].value_counts().plot(
            kind="pie", ax=axes[0, 2], autopct="%1.1f%%",
            colors=["#ff9999", "#66b3ff"], startangle=90,
        )
        axes[0, 2].set_title("Randomization Strategy Mix", fontweight="bold")
        axes[0, 2].set_ylabel("")

        axes[0, 3].plot(df["Task_ID"], df["Utilization"], color="#7f7f7f", alpha=0.7, linewidth=1.5)
        axes[0, 3].set_title("Resource Utilization", fontweight="bold")
        axes[0, 3].set_xlabel("Task ID")
        axes[0, 3].set_ylabel("Fraction of Busy Nodes")
        axes[0, 3].set_ylim(0, 1.1)

        axes[1, 0].plot(df["Task_ID"], df["Latency"], color="purple", linewidth=1.5)
        axes[1, 0].axhline(y=1.5, color="orange", linestyle="--", label="Target <1.5s", alpha=0.8)
        axes[1, 0].set_title("Avg Queue Delay (Wq)", fontweight="bold")
        axes[1, 0].set_xlabel("Task ID")
        axes[1, 0].set_ylabel("Seconds")
        axes[1, 0].legend(fontsize=10)

        axes[1, 1].plot(df["Task_ID"], df["Throughput"], color="#1f77b4", linewidth=2.5)
        axes[1, 1].axhline(y=0.30, color="green", linestyle="--", label="Target ~0.30", alpha=0.8)
        axes[1, 1].set_title("System Throughput", fontweight="bold")
        axes[1, 1].set_xlabel("Task ID")
        axes[1, 1].set_ylabel("Tasks/Sec")
        axes[1, 1].legend(fontsize=10)

        if self.xy_pairs:
            xs = np.array([p[0] for p in self.xy_pairs])
            ys = np.array([p[1] for p in self.xy_pairs])
            rho, _ = stats.pearsonr(xs, ys)
        else:
            xs     = np.random.uniform(0.1, 1.0, 200)
            ys     = np.maximum(0.1, 5 * xs + np.random.normal(0.5, 1.95, 200))
            rho, _ = stats.pearsonr(xs, ys)

        sns.regplot(
            x=xs, y=ys, ax=axes[1, 2],
            scatter_kws={"alpha": 0.4, "s": 15},
            line_kws={"color": "red", "linewidth": 2},
        )
        axes[1, 2].set_title(f"Joint PDF f(X,Y)  ρ ≈ {rho:+.2f}", fontweight="bold")
        axes[1, 2].set_xlabel("Task Size X (MB)")
        axes[1, 2].set_ylabel("CPU Cycles Y")
        axes[1, 2].text(0.05, 0.92, "f(X,Y) ≠ f(X)·f(Y)",
                        transform=axes[1, 2].transAxes, fontsize=10, color="darkred", style="italic")

        axes[1, 3].plot(df["Task_ID"], df["Fairness"], color="orange", linewidth=3)
        axes[1, 3].axhline(y=0.60, color="green", linestyle="--", label="Target ≥0.60", alpha=0.8)
        axes[1, 3].set_title("Jain's Fairness Index", fontweight="bold")
        axes[1, 3].set_xlabel("Task ID")
        axes[1, 3].set_ylabel("Fairness Index J")
        axes[1, 3].set_ylim(0, 1.1)
        axes[1, 3].legend(fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("../../experiments/M4_Final_Dashboard.png", dpi=300, bbox_inches="tight")
        print("\n  Dashboard saved → experiments/M4_Final_Dashboard.png")
        plt.show()

        os.makedirs("../../data", exist_ok=True)
        df.to_csv("../../data/m4_episode_results.csv", index=False)
        if self.xy_pairs:
            xy_df = pd.DataFrame(self.xy_pairs, columns=["Task_Size_X_MB", "CPU_Cycles_Y"])
            xy_df.to_csv("../../data/m4_xy_pairs.csv", index=False)
        print("  Data saved → data/m4_episode_results.csv")
        print("  Data saved → data/m4_xy_pairs.csv")

        explore_pct = (df["Decision"] == "Explore (Randomized)").mean() * 100
        exploit_pct = 100.0 - explore_pct
        print(f"\n{'='*60}")
        print(f"  M4 ALIGNMENT REPORT")
        print(f"{'='*60}")
        print(f"  rho(X,Y) measured    = {rho:+.4f}   target ~+0.55")
        print(f"  Explore fraction     = {explore_pct:5.1f}%    target ~12.6%")
        print(f"  Exploit fraction     = {exploit_pct:5.1f}%    target ~87.4%")
        print(f"  Final TCR            = {df['Success_Rate'].iloc[-1]:5.1f}%    target >80%")
        print(f"  Final Tail (95th)    = {df['Tail'].iloc[-1]:5.2f}s   target <6s")
        print(f"  Final Wq (avg)       = {df['Latency'].iloc[-1]:5.3f}s   target <1.5s")
        print(f"  Final Jain index     = {df['Fairness'].iloc[-1]:5.3f}    target >=0.60")
        print(f"  Final Throughput     = {df['Throughput'].iloc[-1]:5.3f} t/s target ~0.30")
        print(f"{'='*60}")


if __name__ == "__main__":
    sim = M4_Simulation(num_devices=5)
    sim.train()