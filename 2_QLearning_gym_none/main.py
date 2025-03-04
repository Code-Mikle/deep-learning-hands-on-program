import gym
import numpy as np


# Q-learning训练过程
def train(_epsilon, _epsilon_decay, _min_epsilon, _alpha, _gamma, _Q, _episodes):
    for episode in range(_episodes):
        state, info = env.reset()
        # 处理不同版本Gym的reset返回值
        if isinstance(state, tuple):
            state = state

        done = False
        total_reward = 0

        while not done:
            # ε-greedy策略选择动作
            if np.random.rand() < _epsilon:
                action = env.action_space.sample()  # 随机探索
            else:
                action = np.argmax(_Q[state])  # 利用已知信息

            # 执行动作
            result = env.step(action)
            # 处理不同版本Gym的step返回值
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated

            # 计算目标Q值
            if done:
                target = reward
            else:
                target = reward + _gamma * np.max(_Q[next_state])

            # 更新Q表
            _Q[state][action] += _alpha * (target - _Q[state][action])

            state = next_state
            total_reward += reward

        # 衰减探索率
        _epsilon = max(_min_epsilon, _epsilon * _epsilon_decay)

        # 每100回合显示训练进度
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

        # 保存Q-table
        np.save('experiments/q_table_1.npy', _Q)


# 输出Q表的内容
def show_Q_table(_Q):
    # print('Q:',_Q)
    count = 0
    for i in range(n_states):
        max_value_action = _Q[i].argmax()
        if max_value_action == 0:
            print('⬆️', end='')
        elif max_value_action == 1:
            print('➡️', end='')
        elif max_value_action == 2:
            print('⬇️', end='')
        elif max_value_action == 3:
            print('⬅️', end='')
        count += 1
        if count % 12 == 0:
            count = 0
            print()


# 测试训练结果
def test(_Q):
    # 创建测试环境（带渲染）
    test_env = gym.make('CliffWalking-v0', render_mode='human')

    state, info = test_env.reset()
    if isinstance(state, tuple):
        state = state
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(_Q[state])  # 完全贪婪策略
        result = test_env.step(action)

        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated

        total_reward += reward
        state = next_state

    print(f"Final Test Total Reward: {total_reward}")
    env.close()
    test_env.close()


if __name__ == '__main__':
    # 创建训练环境（不带渲染以提高速度）
    env = gym.make('CliffWalking-v0')

    # 参数设置
    alpha = 0.1  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 1.0  # 初始探索率
    epsilon_decay = 0.995  # 探索率衰减率
    min_epsilon = 0.01  # 最小探索率
    episodes = 1000  # 训练回合数

    # 初始化Q表
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    train(epsilon, epsilon_decay, min_epsilon, alpha, gamma, Q, episodes)
    # Q = np.load('experiments/q_table.npy')
    show_Q_table(Q)
    test(Q)
