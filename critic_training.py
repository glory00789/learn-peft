 q"""
PPO Critic Training Demo
用 "I love paris" 的例子，从零训练一个 Critic 网络
观察 V(s) 如何通过 TD bootstrapping 逐步学会传播 reward
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# ============================================================
# 1. 数据定义
# ============================================================

vocab = {
    "<pad>": 0, "I": 1, "love": 2, "paris": 3,
    "also": 4, "great": 5, "what": 6, "else": 7,
    "you": 8, "nice": 9, "to": 10, "meet": 11,
}
VOCAB_SIZE = len(vocab)

# prompt = "I love paris" → token ids [1, 2, 3]
# 每条轨迹包含：
#   states:  s_0(prompt) → s_1 → s_2 → s_3 → s_4(terminal)
#   rewards: r_0, r_1, r_2, r_3  （只有最后一步有 reward）

trajectories = [
    {
        "name": "I also love paris",
        "tokens": ["I", "also", "love", "paris"],
        "states": [
            [1, 2, 3],              # s0: "I love paris"
            [1, 2, 3, 1],           # s1: + "I"
            [1, 2, 3, 1, 4],        # s2: + "also"
            [1, 2, 3, 1, 4, 2],     # s3: + "love"
            [1, 2, 3, 1, 4, 2, 3],  # s4: terminal
        ],
        "rewards": [0.0, 0.0, 0.0, 1.1],
    },
    {
        "name": "great what else you",
        "tokens": ["great", "what", "else", "you"],
        "states": [
            [1, 2, 3],
            [1, 2, 3, 5],
            [1, 2, 3, 5, 6],
            [1, 2, 3, 5, 6, 7],
            [1, 2, 3, 5, 6, 7, 8],
        ],
        "rewards": [0.0, 0.0, 0.0, 0.9],
    },
    {
        "name": "nice to meet you",
        "tokens": ["nice", "to", "meet", "you"],
        "states": [
            [1, 2, 3],
            [1, 2, 3, 9],
            [1, 2, 3, 9, 10],
            [1, 2, 3, 9, 10, 11],
            [1, 2, 3, 9, 10, 11, 8],
        ],
        "rewards": [0.0, 0.0, 0.0, 0.1],
    },
]


# ============================================================
# 2. Critic 网络
#    输入：token 序列（当前状态的文本）
#    输出：V(s)，一个标量
# ============================================================

class Critic(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token_ids: list[int]) -> torch.Tensor:
        x = torch.tensor(token_ids).unsqueeze(0)   # (1, seq_len)
        emb = self.embedding(x).mean(dim=1)         # (1, embed_dim) 平均池化
        return self.mlp(emb).squeeze()              # scalar


# ============================================================
# 3. 打印 V(s) 表格，方便观察训练进度
# ============================================================

def print_v_table(critic: Critic, label: str = ""):
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  {'轨迹':<22} {'token':<8} {'V(s_t)':<10} {'reward'}")
    print(f"  {'─'*50}")
    for traj in trajectories:
        for t, token in enumerate(traj["tokens"]):
            with torch.no_grad():
                v = critic(traj["states"][t]).item()
            r = traj["rewards"][t]
            marker = " ← reward 在这" if r > 0 else ""
            print(f"  {traj['name']:<22} {token:<8} {v:>6.4f}    {r}{marker}")
        print()


# ============================================================
# 4. 训练 Critic
#    用 TD target = r_t + γ·V(s_{t+1}) 作为监督信号
# ============================================================

def train_critic(n_epochs=500, gamma=1.0, lr=0.01):
    # 初始化 Critic 网络，参数随机
    critic = Critic(VOCAB_SIZE)

    # Adam optimizer，只优化 Critic 的参数
    optimizer = optim.Adam(critic.parameters(), lr=lr)

    print_v_table(critic, "训练前（Critic 初始值，接近 0）")

    for epoch in range(n_epochs):
        epoch_loss = 0.0  # 记录这一轮所有 step 的累计 loss

        for traj in trajectories:
            states  = traj["states"]   # 状态列表：s0, s1, s2, s3, s4(terminal)
            rewards = traj["rewards"]  # 每步的 reward：[0, 0, 0, 1.1]
            n_steps = len(rewards)     # = 4（生成了 4 个 token）

            for t in range(n_steps):
                # 判断是否为最后一步（生成最后一个 token 的那步）
                is_terminal = (t == n_steps - 1)

                # 用 Critic 预测当前状态 s_t 的价值 V(s_t)
                # 这是需要被训练的值，requires_grad=True
                v_t = critic(states[t])

                # ── 计算 TD target（Critic 应该预测的"正确答案"）──
                if is_terminal:
                    # 最后一步：下一个状态是 terminal，V(s_terminal) = 0
                    # 所以 target = r_T + γ·0 = r_T（真实 reward 直接进来）
                    td_target = torch.tensor(rewards[t], dtype=torch.float32)
                else:
                    # 中间步：r_t = 0，target = 0 + γ·V(s_{t+1})
                    # 用 no_grad 因为 V(s_next) 只是参考值，不参与梯度计算
                    # （防止梯度通过 target 反向传播，导致训练不稳定）
                    with torch.no_grad():
                        v_next = critic(states[t + 1])
                    td_target = rewards[t] + gamma * v_next
                    # 注意：中间步 rewards[t]=0，所以等价于 td_target = gamma * v_next

                # ── Critic loss：让 V(s_t) 向 td_target 靠近 ──
                # 均方误差：(预测值 - 目标值)²
                # loss 大 → V(s_t) 离 target 远 → 梯度把 V(s_t) 推向 target
                loss = (v_t - td_target) ** 2

                # 清空上一步的梯度（PyTorch 默认累积梯度，每步必须清空）
                optimizer.zero_grad()

                # 反向传播：计算 loss 对 Critic 所有参数的梯度
                loss.backward()

                # 用梯度更新 Critic 参数（Adam 自适应学习率）
                optimizer.step()

                # 累计 loss 用于打印，.item() 把 tensor 转为 Python float
                epoch_loss += loss.item()

        # 打印关键节点的 V 值，观察 reward 是否在逐步向前传播
        if epoch + 1 in (1, 10, 50, 100, 200, 500):
            print_v_table(critic, f"Epoch {epoch+1:4d}  Loss={epoch_loss:.4f}")

    return critic


# ============================================================
# 5. 用训练好的 Critic 计算 GAE
# ============================================================

def compute_gae(critic: Critic, gamma=1.0, lam=0.95):
    print(f"\n{'='*55}")
    print("  GAE 计算结果（训练后）")
    print(f"{'='*55}")

    for traj in trajectories:
        states  = traj["states"]
        rewards = traj["rewards"]
        tokens  = traj["tokens"]
        n = len(rewards)

        # 获取每个状态的 V（包括 terminal V=0）
        V = []
        for t in range(n):
            with torch.no_grad():
                V.append(critic(states[t]).item())
        V.append(0.0)  # V(s_terminal) = 0

        # TD error δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        deltas = [rewards[t] + gamma * V[t+1] - V[t] for t in range(n)]

        # GAE 从后往前累积
        advantages = [0.0] * n
        advantages[-1] = deltas[-1]
        for t in reversed(range(n - 1)):
            advantages[t] = deltas[t] + gamma * lam * advantages[t + 1]

        print(f"\n  轨迹: {traj['name']}")
        print(f"  {'token':<8} {'V(s_t)':<10} {'δ_t':<10} {'Â_t (GAE)'}")
        print(f"  {'─'*45}")
        for t, token in enumerate(tokens):
            print(f"  {token:<8} {V[t]:>7.4f}   {deltas[t]:>+7.4f}   {advantages[t]:>+7.4f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  PPO Critic 训练 Demo")
    print("  prompt: 'I love paris'")
    print("  reward 只在最后一个 token 给出")
    print("=" * 55)

    critic = train_critic(n_epochs=500, gamma=1.0, lr=0.01)
    compute_gae(critic, gamma=1.0, lam=0.95)

    print("\n\n关键观察：")
    print("  1. 训练前 V(s) 接近 0，reward 还没传播")
    print("  2. 早期 epoch：只有最后 token 的 V(s) 动了")
    print("  3. 随着 epoch 增加：reward 像波浪一样往前传")
    print("  4. 最终：高 reward 轨迹的所有 token 都有更高 V(s)")
    print("  5. GAE 给前面的 token 更高 advantage（它们功劳更大）")
