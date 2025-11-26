import os
import time
from argparse import ArgumentParser

from multiprocessing import Process
from random import randint
from statistics import mean
import traceback
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import pickle
import zmq
from pyarrow import deserialize, serialize
from tensorflow.keras.backend import set_session

from model import GDModel
from utils import logger
from config import Config
from utils.data_trans import (create_experiment_dir, find_new_weights,
                              run_weights_subscriber)
from utils.utils import *

parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='localhost',
                    help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000,
                    help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001,
                    help='Learner server port to subscribe model parameters')
parser.add_argument('--exp_path', type=str, default='/data/biaoweilin/guandan/log',
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=4,
                    help='Number of recent checkpoint files to be saved')
parser.add_argument('--observation_space', type=int, default=(567,),
                    help='The YAML configuration file')
parser.add_argument('--action_space', type=int, default=(5, 216),
                    help='The YAML configuration file')

parser.add_argument('--epsilon', type=float, default=0.01 if Config.is_train else 0,
                    help='Epsilon')
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.elu = nn.ELU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        out = self.elu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.elu(out + residual)


class QNetwork(nn.Module):
    def __init__(self, state_dim=513, action_dim=54, hidden_dims=[512, 512, 512]):
        super().__init__()
        input_dim = state_dim + action_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[0]) for _ in range(len(hidden_dims))]
        )
        self.output_layer = nn.Linear(hidden_dims[0], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.input_layer:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


class Player():
    def __init__(self, args) -> None:
        # Set 'allow_growth'
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        # 数据初始化
        self.mb_states_no_action, self.mb_actions, self.mb_rewards, self.mb_q = [], [], [], []
        self.all_mb_states_no_action, self.all_mb_actions, self.all_mb_rewards, self.all_mb_q = [], [], [], []
        self.args = args
        self.step = 0
        self.num_set_weight = 0
        self.send_times = 1

        # 模型初始化
        self.model_id = -1



        # import numpy as np
        # import tensorflow as tf
        # from model import GDModel


        # =======================================================
        # 1. 加载 PyTorch 权重
        # =======================================================
        # print("Loading PyTorch weights...")
        # pth = torch.load("q_network.pth", map_location="cpu")

        # for k, v in pth.items():
        #     print(f"PyTorch: {k} {v.shape}")


        # # =======================================================
        # # 2. 创建 TensorFlow 模型
        # # =======================================================
        # # print("\nBuilding TF model...")
        

        self.model = GDModel(observation_space=567, action_space=(5, 216))
        
        # sess = self.model.sess

        # tf_vars = {v.name: v for v in tf.compat.v1.global_variables()}

        # # print("\nTF variables:")
        # # for name, var in tf_vars.items():
        #     # print(f"TF: {name} {var.shape}")


        # # =======================================================
        # # 3. PyTorch → TF 名字映射表
        # # =======================================================
        # mapping = {
        #     "input_layer.0.weight": "0/v/QNetwork/input_layer/kernel:0",
        #     "input_layer.0.bias":   "0/v/QNetwork/input_layer/bias:0",
        #     "output_layer.weight":  "0/v/QNetwork/output_layer/kernel:0",
        #     "output_layer.bias":    "0/v/QNetwork/output_layer/bias:0",
        # }

        # # residual blocks
        # for i in range(3):
        #     mapping[f"residual_blocks.{i}.fc1.weight"] = f"0/v/QNetwork/res_block_{i}/fc1/kernel:0"
        #     mapping[f"residual_blocks.{i}.fc1.bias"]   = f"0/v/QNetwork/res_block_{i}/fc1/bias:0"
        #     mapping[f"residual_blocks.{i}.fc2.weight"] = f"0/v/QNetwork/res_block_{i}/fc2/kernel:0"
        #     mapping[f"residual_blocks.{i}.fc2.bias"]   = f"0/v/QNetwork/res_block_{i}/fc2/bias:0"


        # # =======================================================
        # # 4. 参数赋值
        # # =======================================================
        # assign_ops = []

        # for pt_name, tf_name in mapping.items():
        #     pt_tensor = pth[pt_name].cpu().numpy()

        #     var = tf_vars[tf_name]

        #     # PyTorch Dense 权重需要转置
        #     if "weight" in pt_name:
        #         pt_tensor = pt_tensor.T   # [out, in] → [in, out]

        #     # print(f"Assign {pt_name} → {tf_name}, shape: {pt_tensor.shape}")
        #     assign_ops.append(var.assign(pt_tensor))

        # sess.run(assign_ops)
        # print("\nAll weights assigned successfully!")


        # # =======================================================
        # # 5. 计算 explain_variance（可选）
        # # =======================================================
        # def explained_variance(y_true, y_pred):
        #     """
        #     y_true, y_pred: numpy arrays
        #     """
        #     var_y = np.var(y_true)
        #     return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

        with open('q_network.ckpt', 'rb') as f:
            weights = pickle.load(f)
        self.model.set_weights(weights)
        print("Loading weights")

        # 测试 explain_variance
        # print("\nTesting PyTorch vs TensorFlow outputs...")

        # 构造随机输入（567维）
        # np.random.seed(42)
        # test_input = np.random.randn(10, 567).astype(np.float32)
        # torch_model = QNetwork()
        # torch_model.load_state_dict(torch.load("q_network.pth"))
        # torch_model.eval()
        # # ---- PyTorch forward ----
        # torch_input = torch.tensor(test_input)
        # with torch.no_grad():
        #     torch_output = torch_model(torch_input).cpu().numpy()   

        # # ---- TensorFlow forward ----
        # tf_output = self.model.forward(torch_input)

        # print("\n==== Check first layer parameters ====")

        # for key in mapping:
        #     print(key)
        #     # PyTorch 第一层
        #     torch_w = pth[key].numpy()
        #     torch_b = pth[key].numpy()

        #     print("Torch weight shape:", torch_w.shape)
        #     print("Torch bias shape:  ", torch_b.shape)

        #     # TF 第一层
        #     tf_w = sess.run(tf_vars[mapping[key]])
        #     tf_b = sess.run(tf_vars[mapping[key]])

        #     print("TF weight shape:", tf_w.shape)
        #     print("TF bias shape:  ", tf_b.shape)

        #     # 对比数值差异
        #     print("\nWeight diff (Torch.T - TF):", np.max(np.abs(torch_w.T - tf_w)))
        #     print("Bias diff   (Torch - TF):  ", np.max(np.abs(torch_b.T - tf_b)))

        # ---- 计算误差 ----
        # diff = torch_output - tf_output
        # mse = np.mean(diff ** 2)
        # max_abs = np.max(np.abs(diff))

        # def explained_variance(y_true, y_pred):
        #     var_y = np.var(y_true)
        #     if var_y < 1e-12:
        #         return 0.0
        #     return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

        # print(torch_output[:10])
        # print(tf_output[:10])
        # ev = explained_variance(torch_output, tf_output)

        # print("MSE =", mse)
        # print("max_abs_diff =", max_abs)
        # print("explained_variance =", ev)

        # self.model = GDModel(self.args.observation_space, (5, 216))

        # 连接learner
        context = zmq.Context()
        context.linger = 0  # For removing linger behavior
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://localhost:{self.args.data_port}')

        print(args.client_index)

        # log文件
        self.args.exp_path += f'/{args.client_index}'
        create_experiment_dir(self.args, f'{args.client_index}-')
        self.args.ckpt_path = self.args.exp_path / 'ckpt'
        self.args.log_path = self.args.exp_path / 'log'
        self.args.ckpt_path.mkdir(exist_ok=True)
        self.args.log_path.mkdir(exist_ok=True)

        logger.configure(str(self.args.log_path))

        # print("subscriber start")
        # 开模型订阅
        if Config.is_train:
            subscriber = Process(target=run_weights_subscriber, args=(self.args, None))
            subscriber.start()

        # print("subscriber end")
        # 初始化模型
        # print('set weight start')
        # model_init_flag = 0
        # while model_init_flag == 0:
        #     new_weights, self.model_id = find_new_weights(-1, self.args.ckpt_path)
        #     if new_weights is not None:
        #         self.model.set_weights(new_weights)
        #         self.num_set_weight += 1
        #         model_init_flag = 1
        # print('set weight success') 

    def sample(self, state) -> int:
        output = self.model.forward(state['x_batch'])
        if self.args.epsilon > 0 and np.random.rand() < self.args.epsilon:
            action_idx = np.random.randint(0, len(state['legal_actions']))
        else:
            action_idx = np.argmax(output)
        # print(output, output[action_idx])
        q = output[action_idx]
        self.step += 1
        action = state['legal_actions'][action_idx]
        self.mb_states_no_action.append(state['x_no_action'])
        self.mb_actions.append(card2array(action))
        self.mb_q.append(q)
        return action_idx
        
    def update_weight(self):
        new_weights, self.model_id = find_new_weights(self.model_id, self.args.ckpt_path)
        if new_weights is not None:
            self.model.set_weights(new_weights)

    def save_data(self, reward):
        self.mb_rewards = [[reward] for _ in range(len(self.mb_states_no_action))]
        self.all_mb_states_no_action += self.mb_states_no_action
        self.all_mb_actions += self.mb_actions
        self.all_mb_rewards += self.mb_rewards
        self.all_mb_q += self.mb_q

        self.mb_states_no_action = []
        self.mb_rewards = []
        self.mb_actions = []
        self.mb_q = []

    def send_data(self, reward):
        # print("prepare")
        # 调整数据格式并发送
        data = self.prepare_training_data(reward)
        # print("send before")
        self.socket.send(serialize(data).to_buffer())
        # print("send after")
        # print(len(data),data)
        self.socket.recv()

        # 打印log
        if self.send_times % 10 == 0:
            self.send_times = 1
            logger.record_tabular("ep_step", self.step)
            logger.dump_tabular()
        else:
            self.send_times += 1

        # 重置数据存储
        self.step = 0
        self.mb_states_no_action, self.mb_actions, self.mb_rewards, self.mb_q = [], [], [], []
        self.all_mb_states_no_action, self.all_mb_actions, self.all_mb_rewards, self.all_mb_q = [], [], [], []

    def prepare_training_data(self, reward):
        states_no_action = np.asarray(self.all_mb_states_no_action)
        actions = np.asarray(self.all_mb_actions)
        rewards = np.asarray(self.all_mb_rewards)
        q = np.asarray(self.all_mb_q)
        data = [states_no_action, actions, q, rewards]
        name = ['x_no_action', 'action', 'q', 'reward']
        return dict(zip(name, data))


def run_one_player(index, args):
    args.client_index = index
    player = Player(args)
    # if not Config.is_train:
    #     player.update_weight()
    # 初始化zmq
    # print("zmq before")
    # print(index)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{6000+index}')
    # print("zmq after")
    # player.update_weight()

    action_index = 0
    while True:
        # 做动作到获得reward
        state = deserialize(socket.recv())
        if not isinstance(state, int) and not isinstance(state, float) and not isinstance(state, str):
            action_index = player.sample(state)
            socket.send(serialize(action_index).to_buffer())
        else:
            # print("3")
            socket.send(b'none')
            # print("Hello1")
            player.save_data(int(state))
            # print("Hello2")
            player.send_data(state)
            # print("Hello3")
            player.update_weight()


def main():
    # print(f"main()")
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_player(index, *x, **kw)
            # print("run_one_player")
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(players):
                    if _i != index:
                        _p.terminate()
        except Exception as e:
            print(e)
            traceback.print_exc()

    players = []
    if Config.is_train:
        for i in range(16 * 4):
            # print(f'start{i}')
            p = Process(target=exit_wrapper, args=(i, args))
            p.start()
            time.sleep(0.05)
            players.append(p)
    else:
        for i in range(2):
            # print(f'start{i}')
            p = Process(target=exit_wrapper, args=(2 * i, args))
            p.start()
            time.sleep(0.5)
            players.append(p)

    for player in players:
        player.join()

if __name__ == '__main__':
    main()
