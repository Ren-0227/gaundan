import asyncio
import websockets
import json
import random
import logging
import sys
import os
from utils import *
from typing import List, Dict, Any
import traceback
import numpy as np
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
import zmq
import msgpack
import random
import time
import uuid
from multiprocessing import Process

def setup_logger(key):
    """为每个客户端创建独立的日志配置"""
    logger = logging.getLogger(f'test_client_{key}')
    
    # 如果logger已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器，使用key作为文件名
    file_handler = logging.FileHandler(f'test_client_{key}.log')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

RANK = {
    2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8,
    10:9, 11:10, 12:11, 13:12, 14:13
}

def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:     # 级数的情况
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)    # 剩余的牌（0-1阵格式）
        one_hot[num_left_cards] = 1
    return one_hot


def _action_seq_list2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = card2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 216)
    return action_seq_array


def _process_action_seq(sequence, length=20):
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

import pickle

import numpy as np
#import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

def mlp(sizes, activation, output_activation=nn.Identity,use_init=False):
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

def shared_mlp(obs_dim, sizes, activation,use_init=False):  # 分两个叉，一个是过softmax的logits，另一个不过，就是单纯的q(s,a)，这里是前面的共享层
    layers = []
    shapes = [obs_dim] + list(sizes)
    for j in range(len(shapes) - 1):
        act = activation
        if use_init:
            net = nn.Linear(shapes[j], shapes[j+1])
            orthogonal_init(net)
            layers += [net, act()]
        else:
            layers += [nn.Linear(shapes[j], shapes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, legalaction=torch.tensor(list(range(10))).to(torch.float32)):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, legalaction)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, legal_action):
        logits = torch.squeeze(self.logits_net(obs)) - (1 - legal_action) * 1e6
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPQ(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.q_net(obs), -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(512, 512, 512, 512, 256), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space
        self.shared = shared_mlp(obs_dim[1], hidden_sizes, activation, use_init=True)
        self.pi = mlp([hidden_sizes[-1], 128, action_space], activation, use_init=True)  # 输出logits
        self.v = mlp([hidden_sizes[-1], 128, 1], activation, use_init=True)  # 输出q(s,a)


    def step(self, obs, legal_action):
        with torch.no_grad():
            shared_feature = self.shared(obs)
            # print(shared_feature.shape, legal_action.shape)
            logits = torch.squeeze(self.pi(shared_feature)) - (1 - legal_action) * 1e8
            pi = Categorical(logits=logits)
            a = pi.sample()
            logp_a = pi.log_prob(a)  # 该动作的log(pi)

            value = torch.squeeze(self.v(shared_feature), -1)
            #print('value', value.shape)
            v = torch.max(value)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def forward(self, obs, act, legal_action):
        shared_feature = self.shared(obs)
        value = torch.squeeze(self.v(shared_feature), -1)
        logits = torch.squeeze(self.pi(shared_feature)) - (1 - legal_action) * 1e8
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a, value
    
    def act(self, obs):
        return self.step(obs)[0]

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()
    

class MLPQNetwork(nn.Module):
    def __init__(self, observation_space,
                 hidden_sizes=(512, 512, 512, 512, 512), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space

        # build Q function
        self.q = MLPQ(obs_dim, hidden_sizes, activation)

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

class GDTestClient:
    def __init__(self, key):
        """
        初始化测试客户端
        :param key: 玩家唯一key（如a1、b1、a2、b2）
        """
        self.key = key
        # self.uri = f"ws://gd-gs6.migufun.com:23456/{key}"
        self.uri = f"ws://localhost:23456/{key}"
        self.ws = None
        self.model = MLPQNetwork(567)
        # 使用key初始化独立的日志记录器
        self.model.load_state_dict(torch.load('./q_network.pth'))
        self.model.eval()
        self.logger = setup_logger(key)

        self.game_stats = []  # 记录每轮的统计信息
        self.current_round = 0
        self.total_rounds = 0
        self.position = None  # 位置信息将从服务器消息中获取
        self.mypos = 0

        self.is_train = False
        if self.is_train:
            self.epsilon = 0.05
        else:
            self.epsilon = 0

        self.context = zmq.Context()
        self.push = self.context.socket(zmq.PUSH)
        self.push.connect("tcp://127.0.0.1:5557")
        self.wins = 0
        self.games = 0
        self.reset()

    def reset(self):
        self.cards: List[int] = []
        
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.action_order = []
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.over = []
        
    async def connect(self):
        """建立WebSocket连接"""
        try:
            self.ws = await websockets.connect(self.uri)
            self.logger.info(f"玩家{self.key}连接成功")
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            return False

    async def handle_message(self, message: str):
        try:
            data = json.loads(message)
            operation = data.get("operation")
            phase = data.get("phase")
            msg_data = data.get("data", {})

            # self.logger.info(f"收到消息: {message}")

            # print(message)
            if operation == "ping":
                # 处理心跳ping消息，回复pong
                response = {
                    "operation": "pong"
                }
                await self.ws.send(json.dumps(response))
                # self.logger.info(f"回复心跳: {response}")

            elif operation == "Deal":
                # 处理发牌消息
                self.cards = msg_data.get("cards", [])
                # 从消息中获取位置信息
                self.position = msg_data.get("position")
                self.mypos = self.position
                # self.logger.info(f"第{self.current_round + 1}轮 - 收到牌: {self.cards}")

            # 在动作序列中记录动作
            elif operation == "PlayCard":
                # 处理其他玩家的出牌信息
                play_position = msg_data.get("position")
                cards = msg_data.get("cards", [])
                just_play = play_position
                action = card2num(cards[2])
                if play_position != self.mypos:
                    for ele in action:
                        self.other_left_hands[ele] -= 1
                if len(self.over) == 0:    # 如果没人出完牌
                    self.action_order.append(just_play)
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)
                elif len(self.over) == 1:    # 只有一个出完牌的（如果队友也先赢了，就会直接结束）
                    if len(action) > 0 and self.flag == 1: # 第一轮有人接下来了，则顺序没问题
                        self.flag = 2
                        if just_play == (self.over[0] + 3) % 4:     # 是头游的上家接下来的
                            self.action_order.append(just_play)       
                            self.action_seq.append(action)
                            self.history_action[play_position].append(action)
                            self.action_order.append(self.over[0])      # 添加第一个出完牌的玩家的信息
                            self.history_action[self.over[0]].append([-1])
                            self.action_seq.append([-1])
                        else:
                            self.action_order.append(just_play)        # 不是头游的上家接的
                            self.action_seq.append(action)
                            self.history_action[play_position].append(action)
                    elif self.flag == 1 and (just_play + 1) % 4 == self.over[0]:      # 出完牌后全都没接的情况，由出完牌的对家出牌（如0、1、2、3、2）
                        self.flag = 2
                        self.action_order.append(just_play)        # 添加出完牌的上家
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                        self.action_order.append(self.over[0])      # 添加第一个出完牌的玩家的信息
                        self.history_action[self.over[0]].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)      # 添加被跳过出牌的玩家的信息
                        self.history_action[(just_play + 2) % 4].append([])
                        self.action_seq.append([])
                    elif just_play == (self.over[0] + 3) % 4 and self.flag == 2:      # 当第一个出完牌的上家已经出过牌了(过完接风的第一轮后或有人接牌了)
                        self.action_order.append(just_play)        # 继续添加正常的信息
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                        self.action_order.append(self.over[0])      # 添加第一个出完牌的玩家的信息
                        self.history_action[self.over[0]].append([-1])
                        self.action_seq.append([-1])
                    else:
                        self.action_order.append(just_play)        # 继续添加正常的信息
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                elif len(self.over) == 2:   # 可能包含两种情形（0、1和1、0出完情况不一样）
                    if len(action) > 0 and self.flag <= 2:           # 有人接下来的情况
                        if (just_play+1) % 4 not in self.over:          # 下家牌没出完时，正常放过去
                            self.flag = 3        
                            self.action_order.append(just_play)        
                            self.action_seq.append(action)
                            self.history_action[play_position].append(action)    
                        else:
                            self.flag = 3
                            self.action_order.append(just_play)        # 是前二游玩家的上家接牌时
                            self.action_seq.append(action)
                            self.history_action[play_position].append(action)
                            self.action_order.append((just_play + 1) % 4)     # 先出完的肯定是紧挨着的上下家
                            self.history_action[(just_play + 1) % 4].append([-1])
                            self.action_seq.append([-1])
                            self.action_order.append((just_play + 2) % 4)     
                            self.history_action[(just_play + 2) % 4].append([-1])
                            self.action_seq.append([-1])     
                    elif self.flag <= 2 and (just_play+1) % 4 in self.over:     # 接风时全都跳过的情况
                        self.flag = 3
                        self.action_order.append(just_play)        # 添加正常的信息
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)     
                        self.action_order.append((just_play + 1) % 4)     # 先出完的肯定是紧挨着的上下家
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)     
                        self.history_action[(just_play + 2) % 4].append([-1])
                        self.action_seq.append([-1])  
                        if just_play == (self.over[-1] + 2) % 4:  # 0、1情况 (1、0情况不用再加了)
                            self.action_order.append((just_play + 3) % 4)     
                            self.history_action[(just_play + 3) % 4].append([])
                            self.action_seq.append([])                             
                    elif (just_play+1) % 4 in self.over and self.flag == 3: # 没出完牌的一定是上下家关系，当其中一个的下家出完时，就是两个出完的
                        self.action_order.append(just_play)        # 继续添加正常的信息
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                        self.action_order.append((just_play + 1) % 4)     # 先出完的肯定是紧挨着的上下家
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)     
                        self.history_action[(just_play + 2) % 4].append([-1])
                        self.action_seq.append([-1])
                    else:
                        self.action_order.append(just_play)        # 继续添加正常的信息
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)

                self.remaining[just_play] -= len(action)
                if self.remaining[just_play] == 0:
                    self.over.append(just_play)
            # 需要做动作
            elif operation == "RequestAction":
                if self.flag == 0:       # 总共牌减去初始手牌
                    init_hand = card2num(self.cards)
                    for ele in init_hand:
                        self.other_left_hands[ele] -= 1
                    self.flag = 1

                await self.handle_action_request(msg_data)

            # 小局结束
            elif operation == "GameResult":
                # 处理游戏结果
                self.current_round += 1
                round_result = msg_data.get("winList", [])
                my_rank = msg_data.get("rank", 0)
                time_used = msg_data.get("time", 0)
                
                # 记录本轮统计信息
                if self.position is not None:
                    self.game_stats.append({
                        "round": self.current_round,
                        "player_rank": round_result[self.position],  # 使用服务器分配的位置
                        "results": round_result,
                        "time": time_used
                    })
                
                self.logger.info(f"第{self.current_round}轮结束")
                # self.logger.info(f"级牌: {my_rank}")
                self.logger.info(f"完整排名: {round_result}")
                if round_result[self.position] == 1 or round_result[(self.position + 2) % 4] == 1:
                    self.wins += 1
                    self.logger.info(f"win: 我方获胜")
                else:
                    self.logger.info(f"lose: 敌方获胜")
                self.games += 1
                self.logger.info(f"{self.wins = }/{self.games = }")
                if round_result[self.position] == 1 or round_result[(self.position + 2) % 4] == 1:
                    if round_result[self.position] == 2 or round_result[(self.position + 2) % 4] == 2:
                        r = +3
                    if round_result[self.position] == 3 or round_result[(self.position + 2) % 4] == 3:
                        r = +2
                    if round_result[self.position] == 4 or round_result[(self.position + 2) % 4] == 4:
                        r = +1
                elif round_result[self.position] == 2 or round_result[(self.position + 2) % 4] == 2:
                    if round_result[self.position] == 3 or round_result[(self.position + 2) % 4] == 3:
                        r = -1
                    if round_result[self.position] == 4 or round_result[(self.position + 2) % 4] == 4:
                        r = -2
                else:
                    r = -3

                self.logger.info(f"用时: {time_used}秒")
                
                # 准备开始新的一轮
                self.reset()
        
        except json.JSONDecodeError:
            self.logger.error(f"消息解析失败: {message}")
            traceback.print_exc()
        except Exception as e:
            self.logger.error(f"处理消息时出错: {e}")
            traceback.print_exc()

    async def handle_action_request(self, msg_data):
        """
        处理出牌请求，随机选择一个动作
        :param actions: 可选的出牌动作列表
        """
        try:
            actions = msg_data.get("actions", [])

            if not actions or len(actions) == 1:
                # 没有可用动作，选择不出（pass）
                action_index = 0
                # self.logger.info(f"第{self.current_round + 1}轮 - 选择不出")
            else:
                state = self.prepare(msg_data)
                x_batch = torch.from_numpy(state["x_batch"])

                with torch.no_grad():
                    q_preds = self.model(x_batch).squeeze()

                # eps = 0.02  # 可调参数，例如 0.1~0.3

                action_idx = torch.argmax(q_preds).item()

                action = actions[action_idx]
                action_index = action["index"]

                # # 计算 max, min
                # q_max = q_preds.max()
                # q_min = q_preds.min()

                # # 计算阈值
                # threshold = eps * q_min + (1 - eps) * q_max

                # # mask 小于阈值的 q 值
                # masked_q = q_preds.clone()
                # masked_q[masked_q < threshold] = float('-inf')

                # # 对剩下的 Q 值做 softmax，得到动作概率分布
                # probs = F.softmax(masked_q, dim=0)

                # # 如果全是 -inf（极端情况），则退化为均匀采样
                # if torch.isnan(probs).any() or torch.isinf(probs).any():
                #     probs = torch.ones_like(q_preds) / len(q_preds)

                # # 根据概率分布采样动作
                # action_idx = torch.multinomial(probs, 1).item()
                # action = actions[action_idx]
                # action_index = action["index"]


                # self.logger.info(f"{probs = } {q_preds = } {actions = }")
                # self.logger.info(f"第{self.current_round + 1}轮 - 选择出牌: {action.get('action', [])}")

            # 发送选择的动作
            response = {
                "operation": "Action",
                "actionIndex": action_index
            }
            await self.ws.send(json.dumps(response))
            # self.logger.info(f"第{self.current_round + 1}轮 - 发送动作: {response}")

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"处理出牌请求时出错: {e}")

    def proc_universal(self, handCards, cur_rank):
        res = np.zeros(12, dtype=np.int8)

        if handCards[(cur_rank-1)*4] == 0:
            return res

        res[0] = 1
        rock_flag = 0
        for i in range(4):
            left, right = 0, 5
            temp = [handCards[i + j*4] if i+j*4 != (cur_rank-1)*4 else 0 for j in range(5)]
            while right <= 12:
                zero_num = temp.count(0)
                if zero_num <= 1:
                    rock_flag = 1
                    break
                else:
                    temp.append(handCards[i + right*4] if i+right*4 != (cur_rank-1)*4 else 0)
                    temp.pop(0)
                    left += 1
                    right += 1
            if rock_flag == 1:
                break
        res[1] = rock_flag

        num_count = [0] * 13
        for i in range(4):
            for j in range(13):
                if handCards[i + j*4] != 0 and i + j*4 != (cur_rank-1)*4:
                    num_count[j] += 1
        num_max = max(num_count)
        if num_max >= 6:
            res[2:8] = 1
        elif num_max == 5:
            res[3:8] = 1
        elif num_max == 4:
            res[4:8] = 1
        elif num_max == 3:
            res[5:8] = 1
        elif num_max == 2:
            res[6:8] = 1
        else:
            res[7] = 1
        temp = 0
        for i in range(13):
            if num_count[i] != 0:
                temp += 1
                if i >= 1:
                    if num_count[i] == 2 and num_count[i-1] >= 3 or num_count[i] >= 3 and num_count[i-1] == 2:
                        res[9] = 1
                    elif num_count[i] == 2 and num_count[i-1] == 2:
                        res[11] = 1
                if i >= 2:
                    if num_count[i-2] == 1 and num_count[i-1] >= 2 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] == 1 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] >= 2 and num_count[i] == 1:
                        res[10] = 1
            else:
                temp = 0
        if temp >= 4:
            res[8] = 1
        return res

    def prepare(self, message):
        num_legal_actions = len(message['actions'])
        legal_actions = [card2num(i['action'][2]) for i in message['actions']]
        my_handcards = card2array(card2num(message['cards']))   # 自己的手牌,54维
        # print('my_handcards', my_handcards)
        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

        universal_card_flag = self.proc_universal(my_handcards, RANK[message['rank']])     # 万能牌的标志位, 12维
        # print('universal_card_flag', universal_card_flag)
        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)

        other_hands = []       # 其余所有玩家手上剩余的牌，54维
        for i in range(54): 
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        # print(self.mypos, "other handcards: ", other_hands)
        other_handcards = card2array(other_hands)      
        # print('other_handcards', other_handcards)
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

        last_action = []         # 最新的动作，54维
        if len(self.action_seq) > 0:
            last_action = card2array(self.action_seq[-1])
        else:
            last_action = card2array([-1])
        # print(last_action)
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)
        
        last_teammate_action = []               # 队友最后的动作， 54维
        if len(self.history_action[(self.mypos + 2) % 4]) > 0 and (self.mypos + 2) % 4 not in self.over:
            last_teammate_action = card2array(self.history_action[(self.mypos + 2) % 4][-1])
        else:
            last_teammate_action = card2array([-1])
        # print(last_teammate_action)
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        my_action_batch = np.zeros(my_handcards_batch.shape)     # 合法动作，54维
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array(action)

        down_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 1) % 4], 27, 1)   # 下家剩余的牌数， 28维
        
        # print(down_num_cards_left)
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        teammate_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 2) % 4], 27, 1)   # 对家剩余的牌数
        
        # print(teammate_num_cards_left)
        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        up_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 3) % 4], 27, 1)   # 上家剩余的牌数
        
        # print(up_num_cards_left)
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 1) % 4]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 1) % 4]))    # 下家打过的牌， 54维
        else:
            down_played_cards = card2array([])
        
        # print(down_played_cards)
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 2) % 4]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 2) % 4]))    # 对家打过的牌
        else:
            teammate_played_cards = card2array([])
        # print(teammate_played_cards)
        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 3) % 4]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 3) % 4]))    # 上家打过的牌
        else:
            up_played_cards = card2array([])
        # print(up_played_cards)
        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)
 
        self_rank = _get_one_hot_array(RANK[message['rank']], 13, 0)         # 己方当前的级牌，13维
        # print(self_rank)
        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        oppo_rank = _get_one_hot_array(RANK[message['rank']], 13, 0)         # 敌方当前的级牌
        # print(oppo_rank)

        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        cur_rank = _get_one_hot_array(RANK[message['rank']], 13, 0)         # 当前的级牌
        # print(cur_rank)

        cur_rank_batch = np.repeat(cur_rank[np.newaxis, :], num_legal_actions, axis=0)

        x_batch = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                        my_action_batch))
        x_no_action = np.hstack((my_handcards,
                            universal_card_flag,
                            other_handcards,
                            last_action,
                            last_teammate_action,
                            down_played_cards,
                            teammate_played_cards,
                            up_played_cards,
                            down_num_cards_left,
                            teammate_num_cards_left,
                            up_num_cards_left,
                            self_rank,
                            oppo_rank,
                            cur_rank
                            ))
        obs = {
            'x_batch': x_batch.astype(np.float32),
            'legal_actions': legal_actions,
            'x_no_action': x_no_action.astype(np.float32),
        }
        return obs
   
    async def run(self):
        """运行客户端"""
        if not await self.connect():
            return

        try:
            async for message in self.ws:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"连接已关闭，共完成 {self.current_round} 轮游戏")
        except Exception as e:
            self.logger.error(f"运行时出错: {e}")
        finally:
            if self.ws:
                await self.ws.close()

async def main():
    """
    主函数，创建并运行测试客户端
    使用示例：python test_client.py a1
    """
    import argparse
    parser = argparse.ArgumentParser(description='掼蛋游戏测试客户端')
    parser.add_argument('key', type=str, help='玩家唯一key(如a1、b1、a2、b2)')
    args = parser.parse_args()

    client = GDTestClient(args.key)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main()) 