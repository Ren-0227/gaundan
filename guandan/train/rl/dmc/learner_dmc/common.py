import datetime
import time
import warnings
from pathlib import Path

import yaml

from agents import agent_registry
from core import Agent
from models import model_registry
import torch
import tensorflow as tf

def get_agent(args, unknown_args):
    model_cls = model_registry.get(args.model)
    agent_cls = agent_registry.get(args.alg)
    agent = agent_cls(model_cls, args.observation_space, args.action_space, args.agent_config, **unknown_args)

    # # =======================================================
    # # 1. 加载 PyTorch 权重
    # # =======================================================
    # print("Loading PyTorch weights...")
    # pth = torch.load("q_network.pth", map_location="cpu")

    # for k, v in pth.items():
    #     print(f"PyTorch: {k} {v.shape}")


    # # =======================================================
    # # 2. 创建 TensorFlow 模型
    # # =======================================================
    # # print("\nBuilding TF model...")
    
    # tf.compat.v1.reset_default_graph()

    # # self.model = GDModel(observation_space=567, action_space=(5, 216))
    # sess = agent.sess

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


    # # 测试 explain_variance
    # print("\nTesting PyTorch vs TensorFlow outputs...")

    # # 构造随机输入（567维）
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
    # tf_output = agent.forward(torch_input)

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

    # # ---- 计算误差 ----
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
    return agent


def load_yaml_config(args, role_type: str) -> None:
    if role_type not in {'actor', 'learner'}:
        raise ValueError('Invalid role type')

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = None

    if config is not None and isinstance(config, dict):
        if role_type in config:
            for k, v in config[role_type].items():
                if k in args:
                    setattr(args, k, v)
                else:
                    warnings.warn(f"Invalid config item '{k}' ignored", RuntimeWarning)
        args.agent_config = config['agent'] if 'agent' in config else None
    else:
        args.agent_config = None


def save_yaml_config(config_path: Path, args, role_type: str, agent: Agent) -> None:
    class Dumper(yaml.Dumper):
        def increase_indent(self, flow=False, *_, **__):
            return super().increase_indent(flow=flow, indentless=False)

    if role_type not in {'actor', 'learner'}:
        raise ValueError('Invalid role type')

    with open(config_path, 'w') as f:
        args_config = {k: v for k, v in vars(args).items() if
                       not k.endswith('path') and k != 'agent_config' and k != 'config'}
        yaml.dump({role_type: args_config}, f, sort_keys=False, Dumper=Dumper)
        f.write('\n')
        yaml.dump({'agent': agent.export_config()}, f, sort_keys=False, Dumper=Dumper)


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')

    args.exp_path.mkdir()
