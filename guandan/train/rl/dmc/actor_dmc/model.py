# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.backend import get_session




# def combined_shape(length, shape=None):
#     if shape is None:
#         return (length,)
#     return (length, shape) if np.isscalar(shape) else (length, *shape)


# def placeholder(dtype=tf.float32, shape=None):
#     return tf.placeholder(dtype=dtype, shape=combined_shape(None, shape))


# def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
#     for h in hidden_sizes[:-1]:
#         x = tf.layers.dense(x, units=h, activation=activation)
#     return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


# # -------------------------
# # Residual Block
# # -------------------------
# class ResidualBlockTF(object):
#     def __init__(self, dim, name):
#         self.dim = dim
#         self.name = name

#     def __call__(self, x):
#         with tf.variable_scope(self.name):
#             residual = x

#             # FC1
#             out = tf.layers.dense(
#                 x, self.dim,
#                 activation=None,
#                 name="fc1"
#             )
#             out = tf.nn.elu(out)

#             # FC2
#             out = tf.layers.dense(
#                 out, self.dim,
#                 activation=None,
#                 name="fc2"
#             )

#             return tf.nn.elu(out + residual)


# # -------------------------
# # QNetwork
# # -------------------------
# class QNetworkTF(object):
#     def __init__(self, state_dim=513, action_dim=54, hidden_dims=[512, 512, 512], name="QNetwork"):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.input_dim = state_dim + action_dim
#         self.hidden_dim = hidden_dims[0]
#         self.num_blocks = len(hidden_dims)
#         self.name = name

#     def __call__(self, x):
#         with tf.variable_scope(self.name):

#             # Input Layer
#             out = tf.layers.dense(
#                 x,
#                 self.hidden_dim,
#                 activation=tf.nn.relu,
#                 name="input_layer"
#             )

#             # Residual Blocks
#             for i in range(self.num_blocks):
#                 block = ResidualBlockTF(self.hidden_dim, name=f"res_block_{i}")
#                 out = block(out)

#             # Output layer → uniform(-3e-3, 3e-3)
#             output = tf.layers.dense(
#                 out,
#                 1,
#                 activation=None,
#                 name="output_layer"
#             )

#             return output

# class GDModel():
#     def __init__(self, observation_space, action_space, config=None, model_id='0', session=None):
#         with tf.variable_scope(model_id):
#             self.x_ph = placeholder(shape=observation_space)
#             # self.z = placeholder(shape=action_space)
#             # self.zero = placeholder(shape=128)

#         # 输出张量
#         self.values = None

#         # Initialize Tensorflow session
#         if session is None:
#             session = get_session()
#         self.sess = session
        
#         self.scope = model_id
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.model_id = model_id
#         self.config = config

#         # Set configurations
#         if config is not None:
#             self.load_config(config)

#         # Build up model
#         self.build()

#         # Build assignment ops
#         self._weight_ph = None
#         self._to_assign = None
#         self._nodes = None
#         self._build_assign()

#         # Build saver
#         self.saver = tf.train.Saver(tf.trainable_variables())

#         # 参数初始化
#         self.sess.run(tf.global_variables_initializer())    

#     def set_weights(self, weights) -> None:
#         feed_dict = {self._weight_ph[var.name]: weight
#                      for (var, weight) in zip(tf.trainable_variables(self.scope), weights)}
#         self.sess.run(self._nodes, feed_dict=feed_dict)

#     def get_weights(self):
#         return self.sess.run(tf.trainable_variables(self.scope))

#     def save(self, path) -> None:
#         self.saver.save(self.sess, str(path))

#     def load(self, path) -> None:
#         self.saver.restore(self.sess, str(path))

#     def _build_assign(self):
#         self._weight_ph, self._to_assign = dict(), dict()
#         variables = tf.trainable_variables(self.scope)
#         for var in variables:
#             self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
#             self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
#         self._nodes = list(self._to_assign.values())
    
#     def forward(self, x_batch):
#         result = self.sess.run(self.values, feed_dict={self.x_ph: x_batch})
#         # print(result.shape)
#         return result

#     def build(self) -> None:
#         with tf.variable_scope(self.scope):
#             with tf.variable_scope('v'):
#                 qnet = QNetworkTF()
#                 self.values = qnet(self.x_ph)
#         # with tf.variable_scope(self.scope):
#         #     with tf.variable_scope('v'):
#         #         self.values = mlp(self.x_ph, [512, 512, 512, 512, 512, 1], activation='tanh',
#         #                                     output_activation=None)

# # if __name__ == '__main__':
# #     model = GDModel((567,), (5, 216))
# #     with open('/home/luyd/guandan/actor_torch/q_network.ckpt', 'rb') as f:
# #         import pickle
# #         new_weights = pickle.load(f)
# #     model.set_weights(new_weights)
# #     b = np.load("/home/luyd/guandan/actor_ppo/debug128.npy", allow_pickle=True).item()
# #     state = b['x_batch'][7]
# #     info = model.forward(state)
# #     info = info.reshape(-1,)
# #     info = info.argsort()[-10:][::-1].tolist()
# #     print(info)


import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import get_session


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dtype=tf.float32, shape=None):
    return tf.placeholder(dtype=dtype, shape=combined_shape(None, shape))


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


class GDModel():
    def __init__(self, observation_space, action_space, config=None, model_id='0', session=None):
        with tf.variable_scope(model_id):
            self.x_ph = placeholder(shape=observation_space)
            # self.z = placeholder(shape=action_space)
            # self.zero = placeholder(shape=128)

        # 输出张量
        self.values = None

        # Initialize Tensorflow session
        if session is None:
            session = get_session()
        self.sess = session
        
        self.scope = model_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_id = model_id
        self.config = config

        # Set configurations
        if config is not None:
            self.load_config(config)

        # Build up model
        self.build()

        # Build assignment ops
        self._weight_ph = None
        self._to_assign = None
        self._nodes = None
        self._build_assign()

        # Build saver
        self.saver = tf.train.Saver(tf.trainable_variables())

        # 参数初始化
        self.sess.run(tf.global_variables_initializer())    

    def set_weights(self, weights) -> None:
        feed_dict = {self._weight_ph[var.name]: weight
                     for (var, weight) in zip(tf.trainable_variables(self.scope), weights)}
        self.sess.run(self._nodes, feed_dict=feed_dict)

    def get_weights(self):
        return self.sess.run(tf.trainable_variables(self.scope))

    def save(self, path) -> None:
        self.saver.save(self.sess, str(path))

    def load(self, path) -> None:
        self.saver.restore(self.sess, str(path))

    def _build_assign(self):
        self._weight_ph, self._to_assign = dict(), dict()
        variables = tf.trainable_variables(self.scope)
        for var in variables:
            self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
            self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
        self._nodes = list(self._to_assign.values())
    
    def forward(self, x_batch):
        return self.sess.run(self.values, feed_dict={self.x_ph: x_batch})

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('v'):
                self.values = mlp(self.x_ph, [512, 512, 512, 512, 512, 1], activation='tanh',
                                            output_activation=None)

if __name__ == '__main__':
    model = GDModel((567,), (5, 216))
    with open('/home/luyd/guandan/actor_torch/q_network.ckpt', 'rb') as f:
        import pickle
        new_weights = pickle.load(f)
    model.set_weights(new_weights)
    b = np.load("/home/luyd/guandan/actor_ppo/debug128.npy", allow_pickle=True).item()
    state = b['x_batch'][7]
    info = model.forward(state)
    info = info.reshape(-1,)
    info = info.argsort()[-10:][::-1].tolist()
    print(info)