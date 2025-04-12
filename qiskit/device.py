from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import time
from qiskit_algorithms.utils import algorithm_globals

class Device:
    def __init__(self, idx, data, labels, maxiter=10, warm_start=True, initial_point=None):
        self.idx = idx
        self.features = MinMaxScaler().fit_transform(data)
        self.target = labels
        self.maxiter = maxiter
        self.train_score_q4 = 0
        self.test_score_q4 = 0
        self.test_score_q4_1 = 0
        self.training_time = 0
        self.sampler = None
        self.backend = None
        simulator = "fake_manila"

        if simulator == "fake_manila":
            self.backend = FakeManilaV2()
            self.sampler = Sampler(mode=self.backend)
            self.sampler.options.default_shots = 10

        elif simulator == "real_quantum":
            service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_IBM_QUANTUM_TOKEN")
            self.backend = service.least_busy(operational=True, simulator=False)
            self.sampler = Sampler(mode=self.backend)
            self.sampler.options.default_shots = 10

        self.optimizer = COBYLA(maxiter=self.maxiter)
        self.objective_func_vals = []
        self.num_features = self.features.shape[1]
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.target, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=self.num_features, reps=1)
        self.ansatz.measure_all()
        self.warm_start = warm_start
        self.initial_point = np.asarray([0.5] * self.ansatz.num_parameters)
        self.old_params = self.initial_point
        self.new_params = self.initial_point
        self.weights_wp = self.initial_point
        self.weights_p = self.initial_point

        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=1
        )
        print(f"Ansatz qubits before pm.run: {self.ansatz.num_qubits}")
        self.isa_qc_ansatz = pm.run(self.ansatz)
        print(f"Ansatz qubits after pm.run: {self.isa_qc_ansatz.num_qubits}")
        print(f"Feature map qubits before pm.run: {self.feature_map.num_qubits}")
        self.isa_qc_feature_map = pm.run(self.feature_map)
        print(f"Feature map qubits after pm.run: {self.isa_qc_feature_map.num_qubits}")

        self.vqc = VQC(
            sampler=self.sampler,
            feature_map=self.isa_qc_feature_map,
            ansatz=self.isa_qc_ansatz,
            optimizer=self.optimizer,
            callback=self.callback_graph,
            warm_start=True,
            pass_manager=pm
        )

    def get_data(self):
        return self.features

    def get_target(self):
        return self.target

    def set_data(self, data):
        self.features = MinMaxScaler().fit_transform(data)

    def set_target(self, target):
        self.target = target

    def callback_graph(self, weights, obj_func_eval):
        self.objective_func_vals.append(obj_func_eval)

    def training(self, initial_point=None):
        start = time.time()
        self.vqc.fit(self.train_features, self.train_labels)
        self.training_time = time.time() - start
        print(f"*********************DEVICE {self.idx}****************")
        print(f"Training time: {round(self.training_time)} seconds")
        self.train_score_q4 = self.vqc.score(self.train_features, self.train_labels)
        self.test_score_q4 = self.vqc.score(self.test_features, self.test_labels)
        print(f"Quantum VQC on the training dataset: {self.train_score_q4:.2f}")
        print(f"Quantum VQC on the test dataset:     {self.test_score_q4:.2f}")

    def evaluate(self, weights):
        self.vqc.initial_point = weights
        self.test_score_q4_1 = self.vqc.score(self.test_features, self.test_labels)