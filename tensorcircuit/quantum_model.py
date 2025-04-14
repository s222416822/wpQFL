import tensorcircuit as tc
import jax.numpy as jnp
import jax
import optax
from config import no_of_qubits, no_of_classes, readout_mode, k

K = tc.set_backend('jax')


def clf(params, c, k):
    """
    Quantum circuit applying a series of gates.

    Args:
        params: Circuit parameters.
        c: TensorCircuit circuit object.
        k: Number of layers.

    Returns:
        Updated circuit.
    """
    for j in range(k):
        for i in range(no_of_qubits - 1):
            c.cnot(i, i + 1)
        for i in range(no_of_qubits):
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c


def readout(c):
    """
    Compute probabilities from circuit output.

    Args:
        c: TensorCircuit circuit object.

    Returns:
        Probabilities based on readout mode.
    """
    if readout_mode == 'softmax':
        logits = []
        for i in range(no_of_qubits):
            logits.append(jnp.real(c.expectation([tc.gates.z(), [i, ]])))
        logits = jnp.stack(logits, axis=-1) * no_of_qubits
        probs = jax.nn.softmax(logits)
    elif readout_mode == 'sample':
        wf = jnp.abs(c.wavefunction()[:no_of_qubits]) ** 2
        probs = wf / jnp.sum(wf)
    return probs


def loss(params, x, y, k):
    """
    Compute loss for the quantum circuit.

    Args:
        params: Circuit parameters.
        x: Input data.
        y: True labels.
        k: Number of layers.

    Returns:
        Loss value.
    """
    c = tc.Circuit(no_of_qubits, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))


def accuracy(params, x, y, k):
    """
    Compute accuracy for the quantum circuit.

    Args:
        params: Circuit parameters.
        x: Input data.
        y: True labels.
        k: Number of layers.

    Returns:
        Boolean array of correct predictions.
    """
    c = tc.Circuit(no_of_qubits, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)


def pred(params, x, k):
    """
    Predict probabilities for input data.

    Args:
        params: Circuit parameters.
        x: Input data.
        k: Number of layers.

    Returns:
        Predicted probabilities.
    """
    c = tc.Circuit(no_of_qubits, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs


loss = K.jit(loss, static_argnums=[3])
accuracy = K.jit(accuracy, static_argnums=[3])
compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])
pred = K.vmap(pred, vectorized_argnums=[1])