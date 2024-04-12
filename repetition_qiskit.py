import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector

# Register for information bit + 2 repeated
q = QuantumRegister(3, 'q1')
# Register for sydrome bits
q2 = QuantumRegister(2, '|0>')
# Classical register for measuring sydrome bits to
creg = ClassicalRegister(2, "c")

circ = QuantumCircuit(q, q2, creg)
circ.h(q[0])
circ.cx(q[0], q[1])
circ.cx(q[0], q[2])

# Induce bit-flip error
circ.x(q[2])

# Detect errors
circ.cx(q[0], q2[0])
circ.cx(q[1], q2[0])
circ.cx(q[0], q2[1])
circ.cx(q[2], q2[1])

# Draw circuit (.ipynb)
circ.draw()

# Measure the sydrome qubits to the classical register
circ.measure(q2, creg)

# Simulate circuit
simulator = Aer.get_backend('qasm_simulator')
job = execute(circ, simulator, shots=1024)
result = job.result()

# Display results
counts = result.get_counts(circ)
print(counts)
