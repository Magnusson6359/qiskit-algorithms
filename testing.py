import qiskit_algorithms.utils as utils
import qiskit_algorithms.time_evolvers.variational.variational_principles.imaginary_mc_lachlan_principle as mc_lachlan
mc_lachlan.idiot_function() # This is a test function to see if the homemade code is imported correctly
from qiskit_algorithms import VarQITE, TimeEvolutionProblem
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import LinCombEstimatorGradient, LinCombQGT
from qiskit_algorithms.time_evolvers.variational.variational_principles import ImaginaryMcLachlanPrinciple
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import ParityMapper

# Hamiltonian
data = np.loadtxt('qubit_hamil_h4.txt', dtype=str)
paulistrings = data[:,0] 
coefficients = (data[:,1])
coefficients = [complex(coeff) for coeff in coefficients]
paulis = list(zip(paulistrings, coefficients))
#print(paulis)
qubit_op = SparsePauliOp.from_list(paulis)

print(f"Number of qubits: {qubit_op.num_qubits}")
#print(qubit_op)

# Ansatz
nreps = 10

#initial_point = hf_params_h2o_jw(nreps=nreps, perturb=0.00)
#print(initial_point)

ansatz = EfficientSU2(qubit_op.num_qubits, reps=nreps)

init_param_values = {}
for i in range(len(ansatz.parameters)):
    init_param_values[ansatz.parameters[i]] = np.pi / 2
#print(init_param_values)

# UCCSD instead
ansatz = UCCSD(num_spatial_orbitals=8, num_particles=[2,2], qubit_mapper=ParityMapper())
hf_initial_point = HFInitialPoint()
hf_initial_point.ansatz = ansatz
init_param_values = hf_initial_point.to_numpy_array()
#print(init_param_values)

#ansatz.decompose().draw("mpl", style="iqx")

# Estimator
seed = 170
estimator = Estimator(options={'seed': seed})

# The 2 main tunable parameters
total_time=40
num_timesteps = 10

var_principle = ImaginaryMcLachlanPrinciple()
time_evo = TimeEvolutionProblem(qubit_op, 
                                time=total_time, 
                                aux_operators=[qubit_op])
qite = VarQITE(ansatz, init_param_values, 
               variational_principle = var_principle,
               estimator=estimator, 
               num_timesteps=num_timesteps)
result = qite.evolve(time_evo)

with open('h_exp_val.txt', 'w') as f:
    with open("times.txt", "w") as g:
        h_exp_val = np.array([ele[0][0].real for ele in result.observables])
        times = result.times
        print(h_exp_val, file=f)
        print(times, file=g)