# DQN及其变种

## 介绍
包括基础的DQN，和其变种Double DQN、Dueling DQN。

<div style="text-align: center">
    <img src="assets/imgs/frozen_lake.gif" alt="frozen_lake" style="width: 360px">
</div>

## 环境配置

为了运行本项目，请确保安装了以下依赖项：
```
python-3.9.21
pytorch-2.2.2
gym-0.26.1
pygame-2.6.1
matplotlib-3.9.4
numpy-1.26.4
pandas-2.2.3
```

## 模型训练

### DQN

```bash
python DQN.py
```

### Double DQN

```bash
python Double_DQN.py
```

### Dueling DQN

```bash
python Dueling_DQN.py
```

## 测试结果

### DQN

```bash
python DQN.py target=test_model
```
参数：
- target: test_model表示进行测试

待添加测试图表！！！！！！！！！！！！！！！！！！

### Double DQN

```bash
python Double_DQN.py target=test_model
```
参数：
- target: test_model表示进行测试

### Dueling DQN

```bash
python src/Dueling_DQN.py target=test_model
```
参数：
- target: test_model表示进行测试

## 文件结构
```text
2_QLearning_gym_none/
│
├── assets/                  # 资源文件夹
│   └── imgs/                # 图片资源
├── logs/                    # 训练过程日志、结果图表等
│   ├── tensorboard/         # tensorboard 日志
│   ├── model_save/          # 训练好的模型权重
│   └── experiment_results/  # 实验结果
├── src/                     # 源代码
│   ├── DQN.py               # DQN训练
│   ├── Double_DQN.py        # Double_DQN训练
│   └── Dueling_DQN.py       # Dueling_DQN训练
└── README.md                # 项目介绍
```