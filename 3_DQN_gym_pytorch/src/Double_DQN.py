import gym
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import deque
import torch
import random


class DoubleDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DoubleDQN, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # 损失函数
        self.loss_fn = nn.MSELoss()

        # 参数设置
        self.memory = deque(maxlen=2000)  # 队列，最大值设为2000
        self.batch_size = 64  # 从队列中抽取agent信息的批大小
        self.alpha = 0.1  # 学习率
        self.gamma = 0.95  # 折扣因子0.99
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.98  # 探索率衰减率
        self.min_epsilon = 0.1  # 最小探索率
        self.episodes = 1000  # 训练回合数

    # 将agent信息存入队列
    def remember(self, s, a, s_, r, done):
        data = (s, a, s_, r, done)
        self.memory.append(data)

    # 从队列取出batch_size大小的agent信息
    def process_data(self):
        # 从队列中，随机取出一个batch大小的数据
        data = random.sample(self.memory, self.batch_size)
        s = np.array([d[0] for d in data])
        a = [d[1] for d in data]
        s_ = np.array([d[2] for d in data])
        r = [d[3] for d in data]
        done = [d[4] for d in data]

        q1 = self.forward(torch.tensor(s))
        q2 = self.forward(torch.tensor(s_))
        q2_max_a =  q2.argmax(dim=1)
        target = None
        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target = r
            else:
                target = r + self.gamma * q2[i][q2_max_a[i]]

        return q1, target

    def get_action_and_q(self, state):
        x = torch.tensor(state)
        x = x.unsqueeze(dim=0)
        q_action_values = self.forward(x)

        # ε-greedy策略选择动作
        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()  # 随机探索
        else:
            # 利用已知信息
            action = torch.argmax(q_action_values).item()   # 用网络输出结果的最大值的下标作为action的值

        return action, q_action_values


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # 训练过程
    def train_double_dqn(self):
        step_count = 0

        for episode in range(self.episodes):
            state, info = env.reset()

            done = False
            total_reward = 0
            total_loss = []

            self.optimizer.zero_grad()

            while not done:
                action, q_action_values = self.get_action_and_q(state)

                # 执行动作
                result = env.step(action)
                next_state, reward, terminated, truncated, info = result
                total_reward += reward
                step_count += 1

                # 保存s, a, s_, r, done
                self.remember(state, action, next_state, reward, done)
                state = next_state

                # 如果到最终状态，就输出以下成绩
                done = terminated or truncated

                # 如果积累的agent数据足够，就开始更新
                if len(self.memory) > self.batch_size:
                    output, target = self.process_data()
                    loss = self.loss_fn(output, target)
                    loss.backward()
                    self.optimizer.step()

                    total_loss.append(loss)
                    if (step_count + 1) % 5 == 0:
                        # 衰减探索率
                        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)



            # 每100回合显示训练进度
            if (episode + 1) % 100 == 0:
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

    def test_double_dqn(self):
        # 创建测试环境（带渲染）
        test_env = gym.make('CartPole-v1', render_mode='human')
        state, info = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            x_tensor = torch.tensor(state)
            action = torch.argmax(self.forward(x_tensor)).item()  # 完全贪婪策略
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
    env = gym.make('CartPole-v1')

    # create DoubleDQN Object
    q_network = DoubleDQN(env.observation_space.shape[0], env.action_space.n)

    q_network.train_double_dqn()
    q_network.test_double_dqn()
