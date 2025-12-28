import numpy as np
import random
from matplotlib import pyplot as plt


# ==========================================
# 1. 环境定义 (Grid World)
# ==========================================
class GridWorld:
    def __init__(self, reward_non_terminal=-0.04):
        self.rows = 3
        self.cols = 4
        self.reward_non_terminal = reward_non_terminal
        # 状态定义: (row, col)
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.blocked_state = (1, 1)  # 灰块
        self.terminal_states = {(2, 3): 1.0, (1, 3): -1.0}  # 终止点 +1 和 -1
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        # self.gamma = 0.99
        self.valid_states = [s for s in self.states if s != self.blocked_state]

    def get_reward(self, state):
        """
        进入 state 时获得的奖励
        :param state:
        :return:
        """
        if state in self.terminal_states:
            return self.terminal_states[state]
        return self.reward_non_terminal

    def transition_probs(self, state, action):
        """返回可能的 (next_state, probability) 列表"""
        if state in self.terminal_states:
            return []  # 终止状态没有转移

        probs = []
        move_map = {'UP': (1, 0), 'DOWN': (-1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
        perpendicular = {
            'UP': ['LEFT', 'RIGHT'], 'DOWN': ['LEFT', 'RIGHT'],
            'LEFT': ['UP', 'DOWN'], 'RIGHT': ['UP', 'DOWN']
        }

        # 0.8 概率执行预期动作，0.1 概率执行垂直方向动作
        for act, p in [(action, 0.8)] + [(a, 0.1) for a in perpendicular[action]]:
            dr, dc = move_map[act]
            next_s = (state[0] + dr, state[1] + dc)
            # 越界或撞墙留在原位
            if next_s not in self.states or next_s == self.blocked_state:
                next_s = state
            probs.append((next_s, p))

        combined = {}
        for s, p in probs:
            combined[s] = combined.get(s, 0) + p
        return list(combined.items())


# ==========================================
# 2. 模型已知算法
# ==========================================
def value_iteration(env, gamma=0.9, epsilon=1e-6, max_iteration=100):
    U = {s: 0.0 for s in env.valid_states}
    history = [U.copy()]
    theta = epsilon * (1 - gamma) / gamma
    for iteration in range(max_iteration):
        delta = 0  # 记录最大变化值
        new_U = U.copy()

        # 遍历所有状态
        for s in env.valid_states:
            # valid_states 中已经避免了墙的出现
            if s in env.terminal_states:
                new_U[s] = env.terminal_states[s]
                continue

            # 计算Q值列表
            action_values = []

            # U'(s) <- R(s) + gamma * max sum P(s'|s,a) * U(s')
            for a in env.actions:
                val = sum(p * U[ns]
                          for ns, p in env.transition_probs(s, a))
                action_values.append(val)
            new_U[s] = env.get_reward(s) + gamma * max(action_values)

            delta = max(delta, abs(new_U[s] - U[s]))

        # 更新值函数
        U = new_U
        history.append(U.copy())

        if delta < theta:
            print(f"\n在第 {iteration} 次迭代后收敛!")
            print(f"最终最大变化: {delta:.6f}")
            break

    # 用 MEU 准则
    # p*(s) = arg max sum(P(s')U*(s'))
    policy = {}
    for s in env.valid_states:
        if s in env.terminal_states: continue
        policy[s] = env.actions[np.argmax([sum(p * U[ns] for ns, p in env.transition_probs(s, a))
                                           for a in env.actions])]
    return U, policy, history


def policy_iteration(env, gamma=0.9, max_iteration=1000):
    U = {s: 0.0 for s in env.valid_states}
    history = [U.copy()]
    policy = {(r, c): random.choice(env.actions)
              for r in range(env.rows)
              for c in range(env.cols) if (r, c) != env.blocked_state and (r, c) not in env.terminal_states}
    for iteration in range(max_iteration):
        U = policy_evaluation(env, policy, U, gamma)
        unchanged = True
        # 遍历所有状态
        for s in env.valid_states:
            # valid_states 中已经避免了墙的出现
            if s in env.terminal_states:
                U[s] = env.terminal_states[s]
                continue

            # max sum P(s'|s,a) * U(s')
            # 计算Q值列表
            action_values = [sum(p * U[ns] for ns, p in env.transition_probs(s, a)) for a in env.actions]
            if max(action_values) > sum(p * U[ns] for ns, p in env.transition_probs(s, policy[s])):
                policy[s] = env.actions[np.argmax(action_values)]
                unchanged = False

        # 将 U 添加到历史
        history.append(U)

        if unchanged:
            print(f"\n在第 {iteration} 次迭代后收敛!")
            break

    return U, policy, history


def policy_evaluation(env, policy, U, gamma=0.9, max_iteration=1000):
    epsilon = 1e-6
    for iteration in range(max_iteration):
        delta = 0  # 记录最大变化值
        new_U = U.copy()
        for s in env.valid_states:
            if s in env.terminal_states:
                new_U[s] = env.terminal_states[s]
                continue
            new_U[s] = env.get_reward(s) + gamma * sum(p * U[ns] for ns, p in env.transition_probs(s, policy[s]))
            delta = max(delta, abs(new_U[s] - U[s]))
        U = new_U
        if delta < epsilon:
            print(iteration)
            break
    return U


# def policy_evaluation(env, policy, U, gamma=0.9):
#     new_U = U.copy()
#     for s in env.valid_states:
#         if s in env.terminal_states:
#             new_U[s] = env.terminal_states[s]
#             continue
#         new_U[s] = env.get_reward(s) + gamma * sum(p * U[ns] for ns, p in env.transition_probs(s, policy[s]))
#     U = new_U
#     return U


# ==========================================
# 3. 模型未知算法 (修正版)
# ==========================================
def exploratory_mc(env: GridWorld, n_trials: int, T: int, epsilon: float):
    # 随机初始化 pi
    pi = {s: random.choice(env.actions) for s in env.valid_states if s not in env.terminal_states}
    # 动作状态效用函数
    Q = {(s, a): 0.0 for s in env.valid_states for a in env.actions}
    # 动作状态频数
    N_sa = {(s, a): 0 for s in env.valid_states if s not in env.terminal_states for a in env.actions}
    # repeat
    for trial in range(n_trials):
        # 探索性开始 (Exploration Start)
        trail = generate_trail(env, pi, T, epsilon)
        for t in range(T):
            R_t = np.mean([trail[i][1] for i in range(t + 1, T + t + 1)])
            s_t = trail[t][0]
            a_t = trail[t][2]
            if s_t in env.terminal_states:
                continue
            Q[(s_t, a_t)] = (Q[(s_t, a_t)] * N_sa[(s_t, a_t)] + R_t) / (N_sa[(s_t, a_t)] + 1)
            N_sa[(s_t, a_t)] += 1
        for s in env.valid_states:
            if s in env.terminal_states:
                continue
            pi[s] = env.actions[np.argmax([Q[(s, a)] for a in env.actions])]
    return pi


def generate_trail(env: GridWorld, pi, T, epsilon):
    trail = []

    curr_s = (0, 0)
    for step in range(2 * T):

        # 如果终止，后续均为终止奖励，强调终止状态的占比
        if curr_s in env.terminal_states:
            reward = env.get_reward(curr_s)
            trail.append((curr_s, reward, None))
            remaining_steps = 2 * T - step
            for _ in range(remaining_steps):
                trail.append((curr_s, reward, None))
            break

        # 当前动作
        curr_a = random.choice(env.actions) if random.random() < epsilon else pi[curr_s]
        # 即时奖励
        curr_reward = env.get_reward(curr_s)
        # 记录 (s, a, r)
        trail.append([curr_s, curr_reward, curr_a])

        # 转移
        if curr_a is None:
            break

        probs = env.transition_probs(curr_s, curr_a)
        if probs:
            next_s = random.choices([p[0] for p in probs], [p[1] for p in probs])[0]
            curr_s = next_s

    return trail


def q_learning_exploration_f(env, n_trials, gamma, R_plus=1, N_e=9):
    # 初始化 Q 值和计数器
    Q = {(s, a): 0.0 for s in env.valid_states for a in env.actions}
    N_sa = {(s, a): 0 for s in env.valid_states if s not in env.terminal_states for a in env.actions}

    # 策略 pi 初始化
    pi = {s: random.choice(env.actions) for s in env.valid_states if s not in env.terminal_states}

    s = None
    # 注意：这里不需要维护全局变量 r，我们直接使用 perceive 返回的即时 r_

    for trial in range(n_trials):
        # 1. 确定当前动作
        if s is None or s in env.terminal_states:
            a = None
        else:
            a = pi[s]

        # 2. 感知环境
        s_, r_ = perceive(env, s, a)  # r_ 是进入 s_ 的奖赏

        # 3. 只有在发生了实际动作 a 时才更新 Q
        if s is not None and a is not None:
            N_sa[(s, a)] += 1
            alpha = 1.0 / (1.0 + 1 * N_sa[(s, a)])  # 动态学习率

            # alpha = 0.1
            # 核心修正：使用本次动作获得的 r_ 进行更新
            max_next_q = max([Q[(s_, a_)] for a_ in env.actions]) if s_ not in env.terminal_states else 0
            Q[(s, a)] += alpha * (r_ + gamma * max_next_q - Q[(s, a)])

            # 4. 更新策略 (使用探索函数)
            # if s_ is not None and s_ not in env.terminal_states:
            pi[s] = env.actions[
                np.argmax([exploration_f(Q[(s, a_)], N_sa.get((s, a_), 0), R_plus, N_e) for a_ in env.actions])]

        # 状态转移
        s = s_

    return pi


def exploration_f(q, n_sa, R_plus, N_e):
    """
    探索函数
    这东西真探索了吗？？？？？？
    :param q:
    :param n_sa:
    :param R_plus:
    :param N_e:
    :return:
    """
    return R_plus if n_sa < N_e else q


def q_learning_epsilon_greedy(env, n_trials=5000, gamma=1, epsilon=0.1):
    # 初始化 Q 值和计数器
    Q = {(s, a): 0.0 for s in env.valid_states for a in env.actions}

    # 策略 pi 初始化
    pi = {s: random.choice(env.actions) for s in env.valid_states if s not in env.terminal_states}

    s = None
    # 注意：这里不需要维护全局变量 r，我们直接使用 perceive 返回的即时 r_

    for trial in range(n_trials):
        # 1. 确定当前动作
        if s is None or s in env.terminal_states:
            a = None
        else:
            a = pi[s] if random.random() > epsilon else random.choice(env.actions)

        # 2. 感知环境
        s_, r_ = perceive(env, s, a)  # r_ 是进入 s_ 的奖赏

        # 3. 只有在发生了实际动作 a 时才更新 Q
        if s is not None and a is not None:
            alpha = 0.1
            # 核心修正：使用本次动作获得的 r_ 进行更新
            max_next_q = max([Q[(s_, a_)] for a_ in env.actions]) if s_ not in env.terminal_states else 0
            Q[(s, a)] += alpha * (r_ + gamma * max_next_q - Q[(s, a)])

            # 4. 更新策略 (使用探索函数)
            pi[s] = env.actions[np.argmax([Q[(s, a_)] for a_ in env.actions])]

        # 状态转移
        s = s_
    return pi


def perceive(env, s, a):
    # 观察
    if s is None or s in env.terminal_states:
        # next_s = random.choice([s for s in env.valid_states if s not in env.terminal_states])
        next_s = (0, 0)
        return next_s, env.get_reward(next_s)
        # return None, 0

    # 转移
    probs = env.transition_probs(s, a)
    next_s = random.choices([p[0] for p in probs], [p[1] for p in probs])[0]
    r = env.get_reward(next_s)
    return next_s, r


# ==========================================
# 4. 执行与展示
# ==========================================
def print_policy_value(env: GridWorld, value, policy, history: list[dict], title):
    print(f"\n--- {title} ---")
    # 为了对应文档坐标图，做简单的 3x4 打印
    res = [["        " for _ in range(4)] for _ in range(3)]
    (r, c) = env.blocked_state
    res[r][c] = "  WALL  "

    for (r, c) in env.terminal_states:
        res[r][c] = f" {env.get_reward((r, c)):.3f} "

    for (r, c), a in policy.items():
        res[r][c] = f"{a[:1]}: {value[(r, c)]:.3f}"  # 简写 U, D, L, R
    for r in range(2, -1, -1):  # 从上往下打印行
        print(f"Row {r + 1}: {res[r]}")

    if history is not None:
        changes = []
        for i in range(1, len(history)):
            max_change = np.max(np.abs(np.array(list(history[i].values())) -
                                       np.array(list(history[i - 1].values()))))
            changes.append(max_change)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(changes) + 1), changes, 'b-', linewidth=2)
        plt.axhline(y=1e-6, color='r', linestyle='--', alpha=0.7, label='Threshold(1e-6)')
        plt.xlabel('Iteration Times', fontsize=12)
        plt.ylabel('Max change', fontsize=12)
        plt.title(title, fontsize=14)
        plt.yscale('log')  # 使用对数坐标更好观察
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def print_policy(env, policy, title):
    print(f"\n--- {title} ---")
    # 为了对应文档坐标图，做简单的 3x4 打印
    res = [[" " for _ in range(4)] for _ in range(3)]
    (r, c) = env.blocked_state
    res[r][c] = "W"

    for (r, c) in env.terminal_states:
        res[r][c] = f" {env.get_reward((r, c)):.0f} "

    for (r, c), a in policy.items():
        if (r, c) not in env.terminal_states:
            res[r][c] = f"{a[:1]}"  # 简写 U, D, L, R
    for r in range(2, -1, -1):  # 从上往下打印行
        print(f"Row {r + 1}: {res[r]}")


if __name__ == "__main__":
    pass
    # # 任务1: 对不同 R(s) 找最优策略 [cite: 6]
    # for rs in [0.01, -0.01, -0.04]:
    #     e = GridWorld(reward_non_terminal=rs)
    #     p_vi = value_iteration(e)
    #     print_policy(p_vi, f"Value Iteration (R={rs})")
    #
    # # 任务2: MC 与 Q-Learning [cite: 7]
    # e_un = GridWorld(reward_non_terminal=-0.04)
    # q_mc = exploratory_mc(e_un)
    # q_ql = q_learning(e_un)
    #
    # # 提取最终策略
    # pol_mc = {s: e_un.actions[np.argmax([q_mc[(s, a)] for a in e_un.actions])]
    #           for s in e_un.valid_states if s not in e_un.terminal_states}
    # pol_ql = {s: e_un.actions[np.argmax([q_ql[(s, a)] for a in e_un.actions])]
    #           for s in e_un.valid_states if s not in e_un.terminal_states}
    #
    # print_policy(pol_mc, "MC Policy")
    # print_policy(pol_ql, "Q-Learning Policy")

    # # 测试世界正确性
    # e = GridWorld(reward_non_terminal=-0.04)
    # probs = e.transition_probs((0, 0), "RIGHT")
    # next_s = random.choices([p[0] for p in probs], [p[1] for p in probs])[0]
    # print(next_s)
    # print(e.get_reward((0, 0)))
    # print(e.get_reward((1, 1)))  # 实际上不会到这个区块
    # print(e.get_reward((2, 3)))
    # print(e.get_reward((1, 3)))
    #
    # print(e.transition_probs((0, 0), "RIGHT"))
    # print(e.transition_probs((0, 0), "LEFT"))
    # print(e.transition_probs((0, 0), "UP"))
    # print(e.transition_probs((0, 0), "DOWN"))
    #
    # print(e.transition_probs((1, 0), "RIGHT"))
    # print(e.transition_probs((1, 0), "LEFT"))
    # print(e.transition_probs((1, 0), "UP"))
    # print(e.transition_probs((1, 0), "DOWN"))

    # 测试值迭代算法
    # for rs in [0.01, -0.01, -0.04]:
    #     e = GridWorld(reward_non_terminal=rs)
    #     u, p, h = value_iteration(e, gamma=1)
    #     print_policy_value(e, u, p, h, title=f"Value Iteration (R={rs}), gamma={1}")

    # 测试策略迭代
    # for rs in [0.01, -0.01, -0.04]:
    #     e = GridWorld(reward_non_terminal=rs)
    #     u, p, h = policy_iteration(e, gamma=1)
    #     print_policy_value(e, u, p, h, title=f"Policy Iteration (R={rs}), gamma={1}")

    # # 测试 MC
    # e_un = GridWorld(reward_non_terminal=-0.04)
    # pol_mc = exploratory_mc(
    #     env=e_un,
    #     n_trials=5000,
    #     T=50,
    #     epsilon=0.1
    # )
    # print_policy(e_un, pol_mc, "MC Policy")

    # 任务2: MC 与 Q-Learning
    e_un = GridWorld(reward_non_terminal=-0.04)
    # q_ql = q_learning_exploration_f(
    #     env=e_un,
    #     n_trials=100000,
    #     gamma=1)
    q_ql = q_learning_epsilon_greedy(
        env=e_un,
        n_trials=100000,
        gamma=1,
        epsilon=0.1
    )
    print_policy(e_un, q_ql, "Q-Learning Policy")
