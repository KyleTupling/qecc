# Import libraries
import random
import numpy as np 
import copy
import time
import matplotlib.pyplot as plt
import csv
import matplotlib 

# Define constants
DEBUG_MODE = False # Enabling this will provide console prints for each step of simulation
CYCLES = 1000 # Simulation run counts

# Enable LaTeX font in plots
font = {'family': 'serif', 'size': 12, 'serif': 'cmr10'}
matplotlib.rc('font', **font)

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# Define quantum basis states
zeroQubit = np.matrix([[1], [0]])
oneQubit = np.matrix([[0], [1]])
negOneQubit = np.matrix([[0], [-1]])

def kronScaleOperator(_operator, _n):
    """
    Scales a single-qubit operator to n-qubit system

    Args:
    _operator (numpy.ndarray): The operator matrix
    _n (int): Number of qubits in system

    Returns:
    numpy.ndarray: The scaled operator
    """
    newOperator = _operator
    for i in range(_n-1):
        newOperator = np.kron(newOperator, I)
    return newOperator

# Define Z operators (scaled to n=9 qubits)
Z1 = np.kron(Z, kronScaleOperator(I, 8))
Z2 = np.kron(I, np.kron(Z, kronScaleOperator(I, 7)))
Z3 = np.kron(kronScaleOperator(I, 2), np.kron(Z, kronScaleOperator(I, 6)))
Z4 = np.kron(kronScaleOperator(I, 3), np.kron(Z, kronScaleOperator(I, 5)))
Z5 = np.kron(kronScaleOperator(I, 4), np.kron(Z, kronScaleOperator(I, 4)))
Z6 = np.kron(kronScaleOperator(I, 5), np.kron(Z, kronScaleOperator(I, 3)))
Z7 = np.kron(kronScaleOperator(I, 6), np.kron(Z, kronScaleOperator(I, 2)))
Z8 = np.kron(kronScaleOperator(I, 7), np.kron(Z, I))
Z9 = np.kron(kronScaleOperator(I, 8), Z)

# Define X operators (scaled to n=9 qubits)
X1 = np.kron(X, kronScaleOperator(I, 8))
X2 = np.kron(I, np.kron(X, kronScaleOperator(I, 7)))
X3 = np.kron(kronScaleOperator(I, 2), np.kron(X, kronScaleOperator(I, 6)))
X4 = np.kron(kronScaleOperator(I, 3), np.kron(X, kronScaleOperator(I, 5)))
X5 = np.kron(kronScaleOperator(I, 4), np.kron(X, kronScaleOperator(I, 4)))
X6 = np.kron(kronScaleOperator(I, 5), np.kron(X, kronScaleOperator(I, 3)))
X7 = np.kron(kronScaleOperator(I, 6), np.kron(X, kronScaleOperator(I, 2)))
X8 = np.kron(kronScaleOperator(I, 7), np.kron(X, I))
X9 = np.kron(kronScaleOperator(I, 8), X)

class ShorState(object):

    def __init__(self, _blockLayoutMat, _blockSignsMat):
        """
        Initialises the Shor state instance using layout schematic parameters. 

        Args:
        _blockLayoutMat (numpy.ndarray): The initial values of the state according to Shor code encoding.
        _blockSignsMat (numpy.ndarray): The signs within each block of the encoded state.
        """
        self.blockLayoutMat = _blockLayoutMat
        self.blockSignsMat = _blockSignsMat

        # Keep track of errors induced
        self.totalErrorCount = 0

        # Encode
        self.statevector = self.encode()

    def __str__(self):
        """
        Format printing of quantum state to be human-readable

        Returns:
        str: Formatted string
        """
        returnStr = ""
        for i, block in enumerate(self.blockLayoutMat):
            returnStr += "("
            returnStr += f" {self.blockSignsMat[i][0]} " if self.blockSignsMat[i][0] == "-" else ""
            for j, subblock in enumerate(block):
                returnStr += "|"
                for k, value in enumerate(subblock):
                    returnStr += str(value)
                returnStr += ">"
                returnStr += f" {self.blockSignsMat[i][1]} " if j % 2 == 0 else ""
            returnStr += ") x " if i < 2 else ") "
                    
        return returnStr
    
    @staticmethod
    def StateBlockDifference(_state1, _state2):
        """
        Calculates how many parameters two Shor-encoded states differ by.
        Includes qubit values (0, 1) as well as qubit signs (+, -).

        Args:
        _state1 (ShorState): One of the states to compare
        _state2 (ShorState): One of the states to compare
        """
        differenceCount = 0
        for i, block in enumerate(_state1.blockLayoutMat):
            for j, subblock in enumerate(block):
                if block[0][j] != _state2.blockLayoutMat[i][0][j]:
                    differenceCount += 1
        
        for i, block in enumerate(_state1.blockSignsMat):
            for j, value in enumerate(block):
                if value != _state2.blockSignsMat[i][j]:
                    differenceCount += 1

        return differenceCount

    def encode(self):
        """
        Converts the state layout schematic into a 2^9 dimension vector, representing the quantum state.

        Returns:
        numpy.ndarray: The encoded quantum state vector
        """
        # Create and store copies of layout and sign matrices so as not to alter the state's values
        blockLayout = copy.deepcopy(self.blockLayoutMat)
        blockSignsMat = copy.deepcopy(self.blockSignsMat)

        # Populate each block with qubits based on instance's schematics
        for i, block in enumerate(blockLayout):
            for j, subblock in enumerate(block):
                for k, value in enumerate(subblock):
                    if value == 0:
                        blockLayout[i][j][k] = zeroQubit
                    elif value == 1:
                        blockLayout[i][j][k] = oneQubit
                    elif value == -1:
                        blockLayout[i][j][k] = negOneQubit

        # Assign signs to each block based on instance's schematics
        encodedBlockLayouts = []
        for i in range(len(blockSignsMat)):
            currentBlock = blockLayout[i]

            if blockSignsMat[i][0] == "+" and blockSignsMat[i][1] == "+":
                encodedBlock = np.kron(np.kron(currentBlock[0][0], currentBlock[0][1]), currentBlock[0][2]) + np.kron(np.kron(currentBlock[1][0], currentBlock[1][1]), currentBlock[1][2])
            elif blockSignsMat[i][0] == "+" and blockSignsMat[i][1] == "-":
                encodedBlock = np.kron(np.kron(currentBlock[0][0], currentBlock[0][1]), currentBlock[0][2]) - np.kron(np.kron(currentBlock[1][0], currentBlock[1][1]), currentBlock[1][2])
            elif blockSignsMat[i][0] == "-" and blockSignsMat[i][1] == "+":
                encodedBlock = np.kron(-1 * np.kron(currentBlock[0][0], currentBlock[0][1]), currentBlock[0][2]) + np.kron(np.kron(currentBlock[1][0], currentBlock[1][1]), currentBlock[1][2])
            else:
                encodedBlock = np.kron(-1 * np.kron(currentBlock[0][0], currentBlock[0][1]), currentBlock[0][2]) - np.kron(np.kron(currentBlock[1][0], currentBlock[1][1]), currentBlock[1][2])

            encodedBlockLayouts.append(encodedBlock)

        return np.kron(encodedBlockLayouts[0], np.kron(encodedBlockLayouts[1], encodedBlockLayouts[2]))
    
    def errorChannel(self, _p):
        """
        The depolarising channel model. Inflicts a Pauli error on the ShorState instance based on depolarising probability.

        Args:
        _p (float): The depolarising probability of the channel
        """
        channelErrorCount = 0

        blockLayout = copy.deepcopy(self.blockLayoutMat)
        blockSignsMat = copy.deepcopy(self.blockSignsMat)

        if DEBUG_MODE: print("-- ERROR CHANNEL --------------------")

        for i, block in enumerate(blockLayout):
            for j in range(3):
                # Bit-flip
                if random.random() <= _p/3:
                    if DEBUG_MODE: print(f"Bits at block {i+1}, index {j + 1} flipping")
                    block[0][j] ^= 1
                    block[1][j] ^= 1
                    channelErrorCount += 1
                # Phase-flip
                elif random.random() <= _p/3: 
                    if DEBUG_MODE: print(f"Phase flipping at block {i + 1}")
                    if block[1][j] == 1:
                        blockSignsMat[i][1] = "-" if blockSignsMat[i][1] == "+" else "+"
                    channelErrorCount += 1
                # Bit and phase flip
                elif random.random() <= _p/3:
                    if DEBUG_MODE: print(f"Bit & Phase flipping at block {i + 1}, index {j + 1}")
                    block[0][j] ^= 1
                    block[1][j] ^= 1
                    blockSignsMat[i][1] = "-" if blockSignsMat[i][1] == "+" else "+"
                    channelErrorCount += 1
                    
        if DEBUG_MODE: print("-------------------------------------")
        
        erroredState = ShorState(blockLayout, blockSignsMat)
        erroredState.totalErrorCount = channelErrorCount
        return erroredState

    def errorCorrection(self):
        """
        Carries out stabiliser operator measurements (expectation values).
        Uses stabiliser eigenvalues to detect and correct errors according to syndrome.
        """
        sv = self.statevector

        Z1Z2_eigenvalue = computeEigenvalue(Z1 @ Z2, sv)
        Z2Z3_eigenvalue = computeEigenvalue(Z2 @ Z3, sv)
        Z4Z5_eigenvalue = computeEigenvalue(Z4 @ Z5, sv)
        Z5Z6_eigenvalue = computeEigenvalue(Z5 @ Z6, sv)
        Z7Z8_eigenvalue = computeEigenvalue(Z7 @ Z8, sv)
        Z8Z9_eigenvalue = computeEigenvalue(Z8 @ Z9, sv)

        if Z1Z2_eigenvalue == -1 and Z2Z3_eigenvalue == 1:
            self.blockLayoutMat[0][0][0] ^= 1
            self.blockLayoutMat[0][1][0] ^= 1
        elif Z1Z2_eigenvalue == -1 and Z2Z3_eigenvalue == -1:
            self.blockLayoutMat[0][0][1] ^= 1
            self.blockLayoutMat[0][1][1] ^= 1
        elif Z1Z2_eigenvalue == 1 and Z2Z3_eigenvalue == -1:
            self.blockLayoutMat[0][0][2] ^= 1
            self.blockLayoutMat[0][1][2] ^= 1

        if Z4Z5_eigenvalue == -1 and Z5Z6_eigenvalue == 1:
            self.blockLayoutMat[1][0][0] ^= 1
            self.blockLayoutMat[1][1][0] ^= 1
        elif Z4Z5_eigenvalue == -1 and Z5Z6_eigenvalue == -1:
            self.blockLayoutMat[1][0][1] ^= 1
            self.blockLayoutMat[1][1][1] ^= 1
        elif Z4Z5_eigenvalue == 1 and Z5Z6_eigenvalue == -1:
            self.blockLayoutMat[1][0][2] ^= 1
            self.blockLayoutMat[1][1][2] ^= 1

        if Z7Z8_eigenvalue == -1 and Z8Z9_eigenvalue == 1:
            self.blockLayoutMat[2][0][0] ^= 1
            self.blockLayoutMat[2][1][0] ^= 1
        elif Z7Z8_eigenvalue == -1 and Z8Z9_eigenvalue == -1:
            self.blockLayoutMat[2][0][1] ^= 1
            self.blockLayoutMat[2][1][1] ^= 1
        elif Z7Z8_eigenvalue == 1 and Z8Z9_eigenvalue == -1:
            self.blockLayoutMat[2][0][2] ^= 1
            self.blockLayoutMat[2][1][2] ^= 1

        PHASE1_eigenvalue = computeEigenvalue(X1 @ X2 @ X3 @ X4 @ X5 @ X6, sv)
        PHASE2_eigenvalue = computeEigenvalue(X4 @ X5 @ X6 @ X7 @ X8 @ X9, sv)

        if PHASE1_eigenvalue == -1 and PHASE2_eigenvalue == 1:
            self.blockSignsMat[0][1] = "+" if self.blockSignsMat[0][1] == "-" else "-"
        elif PHASE1_eigenvalue == -1 and PHASE2_eigenvalue == -1:
            self.blockSignsMat[1][1] = "+" if self.blockSignsMat[1][1] == "-" else "-"
        elif PHASE1_eigenvalue == 1 and PHASE2_eigenvalue == -1:
            self.blockSignsMat[2][1] = "+" if self.blockSignsMat[2][1] == "-" else "-"

        if DEBUG_MODE:
            print(f"Z1 Z2 |psi> eigenvalue: {Z1Z2_eigenvalue}")
            print(f"Z2 Z3 |psi> eigenvalue: {Z2Z3_eigenvalue}")
            print("-------------------------------")
            print(f"Z4 Z5 |psi> eigenvalue: {Z4Z5_eigenvalue}")
            print(f"Z5 Z6 |psi> eigenvalue: {Z5Z6_eigenvalue}")
            print("-------------------------------")
            print(f"Z7 Z8 |psi> eigenvalue: {Z7Z8_eigenvalue}")
            print(f"Z8 Z9 |psi> eigenvalue: {Z8Z9_eigenvalue}")
            print("-------------------------------")
            print(f"X1 X2 X3 X4 X5 X6 |psi> eigenvalue: {PHASE1_eigenvalue}")
            print(f"X4 X5 X6 X7 X8 X9 |psi> eigenvalue: {PHASE2_eigenvalue}")
            print("-------------------------------")

        self.statevector = self.encode()

def computeEigenvalue(_gate, _stateVector):
    """
    Compute the eigenvalue of an operator by calculating expectation value in given state.

    Args:
    _gate (numpy.ndarray): The operator matrix.
    _state_vector (numpy.ndarray): The quantum state vector.

    Returns:
    complex: The eigenvalue associated with the gate.
    """
    # Apply the gate to the state vector
    resultVector = np.dot(_gate, _stateVector)
    
    # Compute the inner product of the resulting state vector with the original state vector
    innerProduct = np.dot(_stateVector.conj().T, resultVector)
    
    # Compute the magnitude of the inner product
    #magnitude = np.abs(innerProduct)
    magnitude = np.dot(_stateVector.conj().T, _stateVector)

    # Compute the eigenvalue
    eigenvalue = innerProduct / magnitude
    
    return eigenvalue

baseBlock = [
    [[0, 0, 0], [1, 1, 1]], 
    [[0, 0, 0], [1, 1, 1]], 
    [[0, 0, 0], [1, 1, 1]]
]

baseBlockSigns = [
    ["+", "+"],
    ["+", "+"],
    ["+", "+"]
]

initTime = time.time()

probabilityList = np.arange(0.005, 0.25, 0.005)
errorList = []
for i in range(len(probabilityList)):
    totalErrors = 0
    for j in range(CYCLES):
        differenceCount = 0
        state = ShorState(baseBlock, baseBlockSigns) # Working
        errorState = state.errorChannel(probabilityList[i])
        errorState.errorCorrection()
        if not np.all(state.statevector == errorState.statevector):
            totalErrors += ShorState.StateBlockDifference(state, errorState)
    errorList.append(totalErrors)
    print(f"Error probability: {probabilityList[i]}")
    print(f"Total errors after {CYCLES} cycles: {totalErrors}")
    print(f"Error rate: {totalErrors / (CYCLES)}")
    print("-----------------------------")

print(f"Time taken: {round(time.time() - initTime, 2)}s")

# Write normalised QBER data to csv file
with open('data/shor_code.csv', 'w') as dataFile:
    writer = csv.writer(dataFile, delimiter=",")
    writer.writerow([error / CYCLES for error in errorList])

# Define data space for simulation data and theoretical model
x = np.linspace(probabilityList[0], probabilityList[-1], 100)
theoreticalX = np.linspace(0, probabilityList[-1], 100)

# Plot theoretical QBER model
y = 1 - ((1-theoreticalX) ** 9) - 9 * theoreticalX * ((1 - theoreticalX) ** 8)
theoreticalPlot = plt.plot(theoreticalX, y, linestyle='dashed', color="black", label="Theoretical model")

# Fit QBER simulation data to curve
z = np.polyfit(probabilityList, [i / (CYCLES) for i in errorList], 3)
f = np.poly1d(z)
y = f(x)

# Plot QBER simulation data
dataPlot = plt.plot(probabilityList, [i / (CYCLES) for i in errorList], 'x', color="blue") 
dataCurvePlot = plt.plot(x, y, color="blue", label="Simulation")

plt.ylim(10**-4, 10**0)
plt.yscale("log")
# plt.title(f"Shor Code Depolarising Probability vs. QBER @{CYCLES} cycles")
plt.xlabel(r"Depolarising probability (p)", fontsize=16)
plt.ylabel(r"QBER", fontsize=16)
plt.legend(loc="lower right", fontsize=14)

plt.show()
