import jax
import optax
import tensorflow as tf
import numpy as np
from config import no_of_qubits, no_of_classes, k


class Device:

    def __init__(self, id, data, params, opt_state):
        self.id = id
        self.data_train = data
        self.cluster = None
        self.old_params = params
        self.params = params
        self.params_p = params
        self.euclidean_list = []
        self.params_list = []
        # self.opt = optax.adam(learning_rate=1e-2)
        self.opt = optax.sgd(learning_rate=0.1)
        self.opt_state = opt_state
        self.train_loss = []
        self.train_list = []


def generate_devices(device_data, batch_size=64):
    key = jax.random.PRNGKey(42)
    devices = []
    opt = optax.sgd(learning_rate=0.1)

    for node, (device_x, device_y) in enumerate(device_data):
        device_id = node
        device_y = jax.nn.one_hot(device_y, no_of_classes)
        # Create batched training data
        data_train = tf.data.Dataset.from_tensor_slices((device_x, device_y)).batch(batch_size)
        key, subkey = jax.random.split(key)
        params = jax.random.normal(subkey, (3 * k, no_of_qubits))
        # Initialize optimizer state
        opt_state = opt.init(params)
        devices.append(Device(device_id, data_train, params, opt_state))
        print(f"Device {node} - Original Classes: {np.unique(device_y.argmax(axis=1))}")

    return devices