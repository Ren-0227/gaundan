import torch
import numpy as np
import tensorflow as tf
from model import GDModel


# =======================================================
# 1. 加载 PyTorch 权重
# =======================================================
print("Loading PyTorch weights...")
pth = torch.load("q_network.pth", map_location="cpu")

for k, v in pth.items():
    print(f"PyTorch: {k} {v.shape}")


# =======================================================
# 2. 创建 TensorFlow 模型
# =======================================================
print("\nBuilding TF model...")
tf.compat.v1.reset_default_graph()

model = GDModel(observation_space=567, action_space=(5, 216))
sess = model.sess

tf_vars = {v.name: v for v in tf.compat.v1.global_variables()}

print("\nTF variables:")
for name, var in tf_vars.items():
    print(f"TF: {name} {var.shape}")


# =======================================================
# 3. PyTorch → TF 名字映射表
# =======================================================
mapping = {
    "input_layer.0.weight": "0/v/QNetwork/input_layer/kernel:0",
    "input_layer.0.bias":   "0/v/QNetwork/input_layer/bias:0",
    "output_layer.weight":  "0/v/QNetwork/output_layer/kernel:0",
    "output_layer.bias":    "0/v/QNetwork/output_layer/bias:0",
}

# residual blocks
for i in range(3):
    mapping[f"residual_blocks.{i}.fc1.weight"] = f"0/v/QNetwork/res_block_{i}/fc1/kernel:0"
    mapping[f"residual_blocks.{i}.fc1.bias"]   = f"0/v/QNetwork/res_block_{i}/fc1/bias:0"
    mapping[f"residual_blocks.{i}.fc2.weight"] = f"0/v/QNetwork/res_block_{i}/fc2/kernel:0"
    mapping[f"residual_blocks.{i}.fc2.bias"]   = f"0/v/QNetwork/res_block_{i}/fc2/bias:0"


# =======================================================
# 4. 参数赋值
# =======================================================
assign_ops = []

for pt_name, tf_name in mapping.items():
    pt_tensor = pth[pt_name].cpu().numpy()

    # 检查 TF 是否存在
    if tf_name not in tf_vars:
        raise ValueError(f"TF variable not found: {tf_name}")

    var = tf_vars[tf_name]

    # PyTorch Dense 权重需要转置
    if "weight" in pt_name:
        pt_tensor = pt_tensor.T   # [out, in] → [in, out]

    print(f"Assign {pt_name} → {tf_name}, shape: {pt_tensor.shape}")
    assign_ops.append(var.assign(pt_tensor))

sess.run(assign_ops)
print("\nAll weights assigned successfully!")


# =======================================================
# 5. 计算 explain_variance（可选）
# =======================================================
def explained_variance(y_true, y_pred):
    """
    y_true, y_pred: numpy arrays
    """
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)


# 测试 explain_variance
print("\nTesting explain_variance...")
fake_y = np.random.randn(1000)
fake_y_pred = fake_y + np.random.randn(1000) * 0.1
ev = explained_variance(fake_y, fake_y_pred)
print("explained_variance =", ev)
