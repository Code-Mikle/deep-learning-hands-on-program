import random

from torch import nn
import torch.nn.functional as F
import gym
import numpy as np
import torch


class DoubleDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DoubleDQN, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')  # 定义环境
    input_dim = env.observation_space.shape[0]
    # print(input_dim, type(input_dim))  # 4 <class 'int'>
    model = DoubleDQN(4, 2)

    state, info = env.reset()
    print(state, type(state))  # [ 0.04613569 -0.00960943  0.04534223 -0.03285744] <class 'numpy.ndarray'>
    result = env.step(1)
    print(result)
    # x = torch.tensor(state)
    # x = x.unsqueeze(0)
    x = torch.randn(4)
    print(x)
    # output = model(x)
    # print(output)
    print(torch.argmax(x).item())
    # res = random.randint(0, 1)
    # print(res)
    # list1 = []
    # for i in range(100):
    #     action = env.action_space.sample()
    #     list1.append(action)
    # print(list1)


