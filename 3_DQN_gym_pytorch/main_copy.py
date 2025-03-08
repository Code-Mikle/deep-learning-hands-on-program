import gym
import numpy as np
from torch import nn
import torch


# 训练过程
def train(_epsilon, _epsilon_decay, _min_epsilon, _alpha, _gamma, _q_network, _episodes, _optimizer):
    grad_accumulation = 4  # 梯度累积步数
    total_loss = 0
    step_count = 0

    for episode in range(_episodes):
        state, info = env.reset()

        done = False
        total_reward = 0

        while not done:
            x = np.asarray(to_one_hot(state, 48), dtype=np.float32)
            x_tensor = torch.tensor(x)
            q_action_values = _q_network(x_tensor)

            # ε-greedy策略选择动作
            if np.random.rand() < _epsilon:
                action = env.action_space.sample()  # 随机探索
            else:
                # 利用已知信息
                action = torch.argmax(q_action_values).item()

            # 执行动作
            result = env.step(action)

            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated

            # 计算目标Q值
            if done:
                target = reward
            else:
                tensor_next_state = torch.tensor(np.asarray(to_one_hot(next_state, 48), dtype=np.float32))
                target = reward + _gamma * _q_network(tensor_next_state).max()

            _optimizer.zero_grad()
            # calculate loss, 更新Q_Network
            loss = nn.MSELoss()(q_action_values.max().unsqueeze(0), torch.tensor([target], dtype=torch.float32))
            loss.backward()

            # 梯度累计
            total_loss += loss.item()
            step_count += 1

            # 控制梯度更新频率
            if step_count % grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(_q_network.parameters(), 1.0)  # 梯度裁剪
                _optimizer.step()
                _optimizer.zero_grad()
                total_loss = 0

            state = next_state
            total_reward += reward

        # 衰减探索率
        _epsilon = max(_min_epsilon, _epsilon * _epsilon_decay)

        # 每100回合显示训练进度
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

          # 保存Q-table
        # np.save('experiments/q_table_1.npy', _Q)

def to_one_hot(i, n_classes=None):
    a = np.zeros(n_classes, dtype=np.int64)
    a[i] = 1
    return a


class Q_Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc3(x)
        return x


# 测试训练结果
def test(_q_network):
    # 创建测试环境（带渲染）
    test_env = gym.make('CliffWalking-v0', render_mode='human')

    state, info = test_env.reset()

    done = False
    total_reward = 0

    while not done:
        x = np.asarray(to_one_hot(state, 48), dtype=np.float32)
        x_tensor = torch.tensor(x)
        action = torch.argmax(_q_network(x_tensor)).item()  # 完全贪婪策略
        result = test_env.step(action)


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

    # create Q_Network Object
    q_network = Q_Network(env.observation_space.n, env.action_space.n)

    # s, info= env.reset()
    # print(s)
    # index = to_one_hot(s, 16)
    # print(index, type(index))
    # index_tensor = torch.tensor(index).float()
    # print(index_tensor, type(index_tensor))
    # res = q_network(index_tensor)
    # print(res, type(res))

    # loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)


    # 参数设置
    alpha = 0.1  # 学习率
    gamma = 0.95  # 折扣因子0.99
    epsilon = 1.0  # 初始探索率
    epsilon_decay = 0.98  # 探索率衰减率
    min_epsilon = 0.1  # 最小探索率
    episodes = 1000  # 训练回合数



    train(epsilon, epsilon_decay, min_epsilon, alpha, gamma, q_network, episodes, optimizer)

    test(q_network)
