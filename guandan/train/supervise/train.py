import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pprint import pprint
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import traceback
import pickle

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def mlp(sizes, activation, output_activation=nn.Identity, use_init=False):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if use_init:
            net = nn.Linear(sizes[j], sizes[j+1])
            orthogonal_init(net)
            layers += [net, act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPQ(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.q_net(obs), -1) # Critical to ensure q has right shape.


class QNetwork(nn.Module):
    def __init__(self, observation_space=567,
                 hidden_sizes=(512, 512, 512, 512, 512), activation=nn.Tanh):
        super().__init__()

        # build Q function
        self.q = MLPQ(observation_space, hidden_sizes, activation)

    def load_tf_weights(self, weights):
        name = ['q_net.0.weight', 'q_net.0.bias', 'q_net.2.weight', 'q_net.2.bias', 'q_net.4.weight', 'q_net.4.bias', 'q_net.6.weight', 'q_net.6.bias', 'q_net.8.weight', 'q_net.8.bias', 'q_net.10.weight', 'q_net.10.bias']
        tensor_weights = []
        for weight in weights:
            temp = torch.tensor(weight).T
            tensor_weights.append(temp)
        new_weights = dict(zip(name, tensor_weights))
        self.q.load_state_dict(new_weights)
        print('load tf weights success')
    
    def forward(self, x):
        return self.q(x)

# 数据集类
class QDataset(Dataset):
    def __init__(self, data_dict):
        """
        data_dict是一个字典，包含's', 'a', 'r', 'q'键对应的数组
        """
        # 直接从字典中获取数据并转换为tensor
        self.states = torch.FloatTensor(data_dict['s'])
        self.actions = torch.FloatTensor(data_dict['a'])  
        self.actions_str = data_dict['a_str']
        self.rewards = torch.FloatTensor(data_dict['r'])
        self.q_values = torch.FloatTensor(data_dict['q'])
        
        print(f"数据集大小: {len(self.states)}")
        print(f"状态维度: {self.states.shape}")
        print(f"动作维度: {self.actions.shape}")
        print(f"奖励维度: {self.rewards.shape}")
        print(f"Q值维度: {self.q_values.shape}")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            's': self.states[idx],
            'a': self.actions[idx],
            'a_str': self.actions_str[idx],
            'r': self.rewards[idx],
            'q': self.q_values[idx]
        }


def explained_variance(y_true, y_pred):
    """
    计算explained variance
    explained_variance = 1 - Var(y_true - y_pred) / Var(y_true)
    """
    var_y = torch.var(y_true)
    var_residual = torch.var(y_true - y_pred)
    if var_y == 0:
        return 0.0
    return 1 - (var_residual / var_y)


def evaluate_model(network, data_loader, max_samples=100000):
    """
    评估模型在验证集上的表现
    """
    network.eval()
    all_q_preds = []
    all_q_targets = []
    total_q_samples = 0

    all_r_preds = []
    all_r_targets = []
    total_r_samples = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            s, a, q_target, r_target = batch_data['s'], batch_data['a'], batch_data['q'], batch_data['r']
            
            # 拼接状态和动作
            x = torch.cat([s, a], dim=-1)
            
            # 预测
            q_pred = network(x)
            
            all_q_preds.append(q_pred)
            all_q_targets.append(q_target)

            total_q_samples += len(q_pred)

            r_pred = network(x)
            all_r_preds.append(r_pred)
            all_r_targets.append(r_target)
            
            total_r_samples += len(r_pred)
            if total_q_samples >= max_samples:
                break
    
    # 合并所有batch的结果
    all_q_preds = torch.cat(all_q_preds, dim=0)[:max_samples]
    all_q_targets = torch.cat(all_q_targets, dim=0)[:max_samples]

    all_r_preds = torch.cat(all_r_preds, dim=0)[:max_samples]
    all_r_targets = torch.cat(all_r_targets, dim=0)[:max_samples]
    
    # 计算指标
    mse = F.mse_loss(all_q_preds, all_q_targets).item()
    mae = F.l1_loss(all_q_preds, all_q_targets).item()

    rho, _ = spearmanr(all_r_preds, all_r_targets)
    pear, _ = pearsonr(all_r_preds, all_r_targets)
    exp_var = explained_variance(all_q_targets, all_q_preds).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'explained_variance': exp_var,
        'rho': rho,
        'pear': pear,
        'q_pred_mean': all_q_preds.mean().item(),
        'q_pred_std': all_q_preds.std().item(),
        'q_pred_min': all_q_preds.min().item(),
        'q_pred_max': all_q_preds.max().item(),
        'q_target_mean': all_q_targets.mean().item(),
        'q_target_std': all_q_targets.std().item(),
        'r_pred_mean': all_r_preds.mean().item(),
        'r_pred_std': all_r_preds.std().item(),
        'r_pred_min': all_r_preds.min().item(),
        'r_pred_max': all_r_preds.max().item(),
        'r_target_mean': all_r_targets.mean().item(),
        'r_target_std': all_r_targets.std().item(),
        'num_samples': len(all_q_preds),
    }


# 加载数据
import glob

file_list = ["1.npy", "2.npy", "3.npy", "5.npy", "6.npy", "7.npy", "8.npy", "9.npy", "10.npy", "11.npy", "12.npy", "13.npy", "14.npy", "15.npy"]

train_data = {"s": [], "a": [], "a_str": [], "r": [], "q": []}

for f in file_list:
    d = np.load(f, allow_pickle=True).item()  # 读取成 dict
    train_data["s"].append(d["s"])
    train_data["a"].append(d["a"])
    train_data["a_str"].append(d["a_str"])
    train_data["r"].append(d["r"])
    train_data["q"].append(d["q"])

# 拼接为一个整体的 ndarray
train_data["s"] = np.concatenate(train_data["s"], axis=0)
train_data["a"] = np.concatenate(train_data["a"], axis=0)
train_data["a_str"] = np.concatenate(train_data["a_str"], axis=0)
train_data["r"] = np.concatenate(train_data["r"], axis=0)
train_data["q"] = np.concatenate(train_data["q"], axis=0)

# 统计训练集 Q 值分布信息
q_values = train_data["q"]
print("\n========== 训练集 Q 值统计 ==========")
print(f"样本数量: {len(q_values)}")
print(f"Q 值范围: [{q_values.min():.4f}, {q_values.max():.4f}]")
print(f"Q 值均值: {q_values.mean():.4f}")
print(f"Q 值标准差: {q_values.std():.4f}")
print(f"Q 值中位数: {np.median(q_values):.4f}")
print("===================================\n")

r_values = train_data["r"]
print("\n========== 训练集 r 值统计 ==========")
print(f"样本数量: {len(r_values)}")
print(f"r 值范围: [{r_values.min():.4f}, {r_values.max():.4f}]")
print(f"r 值均值: {r_values.mean():.4f}")
print(f"r 值标准差: {r_values.std():.4f}")
print(f"r 值中位数: {np.median(r_values):.4f}")
print("===================================\n")

from scipy.stats import spearmanr, pearsonr

q = train_data["q"]
r = train_data["r"]

rho, _ = spearmanr(q, r)
pear, _ = pearsonr(q, r)

print(f"Spearman ρ = {rho:.3f}, Pearson r = {pear:.3f}")

test_data = np.load("./4.npy", allow_pickle=True).item()

# 创建数据集和数据加载器
train_dataset = QDataset(train_data)
test_dataset = QDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=32768, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32768, shuffle=False)
print("数据加载器创建成功\n")

try:
    network = QNetwork()
    # network.load_state_dict(torch.load("q_network.pth"))

    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=1e-5,
        weight_decay=1e-4,
    )
    smooth_loss = nn.SmoothL1Loss(beta=1.0)
    mse_loss = nn.MSELoss()
    print(f"网络结构:")
    print(network)
    print(f"参数数量: {sum(p.numel() for p in network.parameters())}\n")

    # 训练循环
    num_epochs = 10000
    k_epoch = 1  # 每个batch重复训练的次数
    save_every_steps = 50
    eval_every_steps = 50  # 每100步评估一次
    global_step = 0  # 全局步数计数器

    print(f"开始训练... (k_epoch={k_epoch})\n")

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        network.train()  # 设置为训练模式
        
        for batch_idx, batch_data in enumerate(train_data_loader):
            s, a, r, q_ = batch_data['s'], batch_data['a'], batch_data['r'], batch_data['q']
            
            # 拼接状态和动作
            x = torch.cat([s, a], dim=-1)
            
            # 对同一个batch重复训练k_epoch次
            batch_loss = 0.0
            for k in range(k_epoch):
                # 前向传播
                q_pred = network(x)

                # 计算损失
                q_loss = mse_loss(q_pred, q_)
                r_loss = smooth_loss(q_pred, r)

                loss = q_loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_loss += loss.item()
            
            # 计算该batch的平均损失
            avg_batch_loss = batch_loss / k_epoch
            total_loss += avg_batch_loss
            num_batches += 1
            global_step += 1
            
            # 打印训练损失
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step {global_step}, Batch [{batch_idx+1}], Avg Loss: {avg_batch_loss:.6f}")
            
            if global_step % save_every_steps == 0:
                torch.save(network.state_dict(), f'q_network_model_step{global_step+1}.pth')
                print(f"模型已保存为 'q_network_model_step{global_step+1}.pth'")
                print("=" * 70 + "\n")
                
            if global_step % eval_every_steps == 0:
                print("\n" + "=" * 70)
                print(f"Step {global_step} - 在验证集上评估模型（前100000个样本）...")
                eval_metrics = evaluate_model(network, test_data_loader, max_samples=100000)
                
                print(f"验证集评估结果:")
                print(f"  样本数量: {eval_metrics['num_samples']}")
                print(f"  MSE: {eval_metrics['mse']:.6f}")
                print(f"  MAE: {eval_metrics['mae']:.6f}")
                print(f"  Explained Variance: {eval_metrics['explained_variance']:.6f}")
                print(f"  Spearman ρ: {eval_metrics['rho']:.6f}")
                print(f"  Pearson r: {eval_metrics['pear']:.6f}")
                print(f"  预测值 - 均值: {eval_metrics['q_pred_mean']:.4f}, 标准差: {eval_metrics['q_pred_std']:.4f}")
                print(f"  预测值 - 范围: [{eval_metrics['q_pred_min']:.4f}, {eval_metrics['q_pred_max']:.4f}]")
                print(f"  真实值 - 均值: {eval_metrics['q_target_mean']:.4f}, 标准差: {eval_metrics['q_target_std']:.4f}")
                print("=" * 70 + "\n")
                
                network.train()  # 评估后切回训练模式
        
        # 打印每个epoch的平均损失
        avg_loss = total_loss / num_batches
        print(f"\n{'=' * 70}")
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成, 平均训练损失: {avg_loss:.6f}")

    print("训练完成!")
except Exception as e:
    print(f"{e = }")
    traceback.print_exc()
