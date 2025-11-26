# gaundan
以下是基于你提供的代码片段完善的 README 示例，涵盖项目概述、环境配置、核心功能、使用说明等关键信息，供你参考：


# Guandan AI 训练框架

一个用于掼蛋（Guandan）游戏的 AI 训练与测试框架，支持多种强化学习算法（如 PPO、DMC）及监督学习，提供完整的环境配置、数据处理、模型训练和测试流程。


## 项目概述

本项目旨在构建掼蛋游戏的 AI 训练体系，通过强化学习（RL）和监督学习（Supervised Learning）方法训练智能体，实现自动决策。框架包含日志管理、数据预处理、模型训练、环境交互等模块，支持多智能体协作与对抗场景。


## 环境配置

### 依赖安装

项目基于 Python 3.6 开发，依赖通过 Conda 管理，主要包含 TensorFlow、PyTorch、OpenCV、 Gym 等库。

1. 克隆项目并进入目录：
   ```bash
   git clone <项目仓库地址>
   cd guandan
   ```

2. 使用 Conda 创建环境：
   ```bash
   conda env create -f gaundan/guandan/train/rl/dmc/learner_dmc/build/conda/env_linux.yaml
   conda activate framework
   ```


## 项目结构

```
guandan/
├── env/                  # 游戏环境配置
│   ├── linux/
│   │   ├── AIConfig.json  # 游戏参数配置（轮次、超时时间、日志路径等）
│   │   └── gd_client_gui.py  # 游戏界面信息更新
├── train/                # 训练相关代码
│   ├── rl/               # 强化学习模块
│   │   ├── ppo/          # PPO 算法（actor/learner）
│   │   └── dmc/          # DMC 算法（actor/learner）
│   └── supervise/        # 监督学习模块
│       └── train.py      # 监督学习数据集与训练逻辑
├── test/                 # 测试脚本
│   ├── dmc.py            # DMC 算法测试
│   ├── ppo.py            # PPO 算法测试
│   └── sdmc.py           # 其他变体测试
└── utils/                # 工具函数
    ├── logger.py         # 日志管理（记录、打印、存储日志）
    └── data_trans.py     # 数据传输与处理（如模型权重接收）
```


## 核心功能

### 1. 日志管理（logger.py）

提供日志记录、级别控制、键值对存储等功能，支持调试信息输出和训练指标持久化。

示例用法：
```python
from utils.logger import info, debug, set_level, logkv, dumpkvs

info("训练开始")
set_level(DEBUG)  # 开启调试日志
logkv("loss", 0.5)  # 记录键值对
dumpkvs()  # 保存日志到文件
```


### 2. 数据预处理（prepare 方法）

在测试脚本（如 dmc.py、ppo.py）中，`prepare` 方法将游戏状态（手牌、历史动作、剩余牌数等）转换为模型输入特征，包括：
- 手牌编码（54维数组）
- 万能牌标志位（12维）
- 历史动作与玩家状态（上家/下家/队友的牌数、已出牌等）
- 级牌信息（当前级牌、己方/敌方级牌的独热编码）

输出特征用于模型决策，支持批量处理合法动作。


### 3. 模型训练

- **强化学习**：支持 PPO 和 DMC 算法，包含 actor（动作生成）和 learner（模型更新）模块，通过 ZMQ 进行权重传输（data_trans.py）。
- **监督学习**：基于 `QDataset` 类加载数据集（状态、动作、奖励、Q值），实现监督训练流程。


### 4. 游戏环境配置（AIConfig.json）

配置游戏核心参数：
- 总轮次（totalRounds）、超时时间（timeoutSeconds）
- 日志级别（logLevel）与路径（logPath）
- 智能体数量（aiNum）、通信端点（wsEndpoint）
- 级牌规则（rankOption）、重连设置等


## 使用说明

### 1. 启动游戏环境

修改 `env/linux/AIConfig.json` 配置游戏参数，如轮次、日志路径等，然后启动环境客户端。

### 2. 训练模型

- **强化学习训练**：
  ```bash
  # 例如启动 PPO 训练
  python train/rl/ppo/learner_torch/train.py
  ```

- **监督学习训练**：
  ```bash
  python train/supervise/train.py --data_path <数据集路径>
  ```

### 3. 测试模型

运行测试脚本验证模型性能：
```bash
# 测试 DMC 算法
python test/dmc.py

# 测试 PPO 算法
python test/ppo.py
```


## 注意事项

- 确保 Conda 环境依赖安装完整，尤其是 TensorFlow-GPU 和 PyTorch 版本兼容。
- 日志默认存储路径为 `./logs`，可在 `AIConfig.json` 中修改。
- 模型权重通过 ZMQ 传输，需确保 actor 与 learner 通信正常。


## 未来扩展

- 支持更多强化学习算法（如 DQN、A2C）
- 优化数据预处理效率，增加特征维度
- 完善多智能体协作策略
```


你可以根据实际项目细节补充更多内容，如具体算法原理、数据集格式、模型结构等。如果有特定模块需要详细说明，可进一步扩展对应章节。
