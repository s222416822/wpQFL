import time
import os
from datetime import datetime
import optax
import jax.numpy as jnp
from tqdm import tqdm
from quantum_model import compute_loss, compute_accuracy, pred
from config import comm_rounds, dataset_used, datasize_used, no_of_qubits, n_node, n_class, k
from distance_metrics import compute_distances


def main_method(method, g, l, noniid_data, noniid_type, devices_list, x_test, y_test, distance_metrics_used=None):
    """
    Main training method for federated learning.

    Args:
        method: Training method ('Default', 'Weighted', 'Euclidean').
        g: Global model weight.
        l: Local model weight.
        noniid_data: Non-IID data partitions.
        noniid_type: Type of non-IID partitioning.
        devices_list: List of Device instances.
        x_test: Test data features.
        y_test: Test data labels.
        distance_metrics_used: Distance metric for Euclidean method.
    """
    overall_start_time = time.time()
    if distance_metrics_used is None:
        distance_metrics_used = "NotRequired"

    experiment = f"{method}_{datasize_used}_{dataset_used}_{no_of_qubits}q_{n_node}n_{k}k_nClass={n_class}_r={comm_rounds}_{g}g_{l}_metric={distance_metrics_used}_noniid={noniid_type}"
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs = f"logs_tensorcircuit/{date_time}_{experiment}"

    if not os.path.exists(logs):
        os.makedirs(logs)

    def workerTask(logs, device, local_epochs, b):
        print(f"Device {device.id} worker training start...\n")
        data_train_length = len(device.data_train)
        results = []

        for epoch in tqdm(range(local_epochs), leave=False):
            highest_acc = 0
            total_acc = 0
            num_records = 0
            average_acc = 0
            last_acc = 0

            for i, (x, y) in enumerate(device.data_train):
                x, y = x.numpy(), y.numpy()
                device.params = device.params_p
                loss_val, grad_val = compute_loss(device.params, x, y, k)
                updates, device.opt_state = device.opt.update(grad_val, device.opt_state, device.params)
                device.params = optax.apply_updates(device.params, updates)

                if i % 5 == 0:
                    loss_mean = jnp.mean(loss_val)
                    acc = jnp.mean(compute_accuracy(device.params, x, y, k))
                    tqdm.write(
                        f'world {b}, epoch {epoch}, {i}/{data_train_length}: loss={loss_mean:.4f}, acc={acc:.4f}')
                    # with open(f"{logs}/train_results_data_iteration.txt", "a") as file:
                    #   file.write(f'world {b}, epoch {epoch}, {i}/{data_train_length}: loss={loss_mean:.4f}, acc={acc:.4f}')
                    total_acc += acc
                    num_records += 1
                    last_acc = acc
                    if acc > highest_acc:
                        highest_acc = acc

            average_acc = total_acc / num_records
            results.append(
                f"Comm: {b} - Device {device.id} - train_loss: {loss_mean} - highest_train_acc: {highest_acc:.2f} - avg_train_acc: {average_acc:.2f} - last_acc: {last_acc}\n")
            print(
                f"Device {device.id} training Epoch: {epoch} done, Highest Train Acc: {highest_acc:.2f}, Average Train Acc: {average_acc:.2f}")

        try:
            with open(f"{logs}/train_results.txt", "a") as file:
                file.writelines(results)
        except Exception as e:
            print(f"Failed to save results to file: {e}")

        print(f"Device {device.id} work COMPLETE")

    def device_training(logs, local_epochs, b):
        for device in devices_list:
            print(f"=====================Device {device.id} training start...")
            workerTask(logs, device, local_epochs, b)

    def serverTask(b, method, g, l):
        params_list = [device.params for device in devices_list]
        avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)

        for device in devices_list:
            if method == "Default":
                device.old_params = device.params
                device.params_p = avg_params
            elif method == "Weighted":
                device.params_p = (g * avg_params + l * device.params) / (g + l)
            elif method == "Euclidean":

                ews, ewi = compute_distances(device.params, avg_params, device.old_params, distance_metrics_used)
                device.old_params = device.params
                print("Euclidean Distance with Global Model", ews)
                print("Euclidean Distance with Own Old Local Model", ewi)
                if ews < ewi:
                    device.params_p = avg_params
                else:
                    device.params_p = (avg_params + device.params) / 2

        test_acc = jnp.mean(pred(avg_params, x_test[:100], k).argmax(axis=-1) == y_test[:100].argmax(axis=-1))
        test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:100], k)) * y_test[:100])
        tqdm.write(f'Comm. {b}: Server Test Acc={test_acc:.4f}, Server Test Loss={test_loss:.4f}')
        with open(f"{logs}/server_test_results.txt", "a") as file:
            file.write(f"Comm: {b} - test_loss: {test_loss} - test_acc: {test_acc}\n")

    for b in range(comm_rounds):
        current_time = time.time_ns()
        print(f"Communication Round: {b}")
        device_training(logs, 1, b)
        serverTask(b, method, g, l)
        final_time = time.time_ns() - current_time

        with open(f"{logs}/comm_time.txt", "a") as file:
            file.write(f"Comm: {b} - Time: {final_time}\n")

    overall_elapsed_time = time.time() - overall_start_time
    hours, rem = divmod(overall_elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Overall Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))