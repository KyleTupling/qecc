import random
import numpy as np 
import copy
import time
import matplotlib.pyplot as plt

CYCLES = 200

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
I = np.eye(2)

zeroQubit = np.matrix([[1], [0]])
oneQubit = np.matrix([[0], [1]])
negOneQubit = np.matrix([[0], [-1]])

# Scale operator to n qubit system
def kronScaleOperator(_operator, _n):
    newOperator = _operator
    for i in range(_n-1):
        newOperator = np.kron(newOperator, I)
    return newOperator

# Define Z operators (scaled to n=9 qubits)
Z1 = np.kron(Z, kronScaleOperator(I, 6))
Z2 = np.kron(I, np.kron(Z, kronScaleOperator(I, 5)))
Z3 = np.kron(kronScaleOperator(I, 2), np.kron(Z, kronScaleOperator(I, 4)))
Z4 = np.kron(kronScaleOperator(I, 3), np.kron(Z, kronScaleOperator(I, 3)))
Z5 = np.kron(kronScaleOperator(I, 4), np.kron(Z, kronScaleOperator(I, 2)))
Z6 = np.kron(kronScaleOperator(I, 5), np.kron(Z, I))
Z7 = np.kron(kronScaleOperator(I, 6), Z)
ZOperators = [Z1, Z2, Z3, Z4, Z5, Z6, Z7]

# Define X operators (scaled to n=9 qubits)
X1 = np.kron(X, kronScaleOperator(I, 6))
X2 = np.kron(I, np.kron(X, kronScaleOperator(I, 5)))
X3 = np.kron(kronScaleOperator(I, 2), np.kron(X, kronScaleOperator(I, 4)))
X4 = np.kron(kronScaleOperator(I, 3), np.kron(X, kronScaleOperator(I, 3)))
X5 = np.kron(kronScaleOperator(I, 4), np.kron(X, kronScaleOperator(I, 2)))
X6 = np.kron(kronScaleOperator(I, 5), np.kron(X, I))
X7 = np.kron(kronScaleOperator(I, 6), X)
XOperators = [X1, X2, X3, X4, X5, X6, X7]

class SteaneState(object):

    def __init__(self, _blockLayout, _signLayout):
        self.blockLayoutMat = _blockLayout
        self.blockSignsMat = _signLayout

        self.statevector = self.encode()

    def __str__(self):
        returnStr = ""
        for i in range(len(self.blockLayoutMat)):
            returnStr += "|"
            for j in range(len(self.blockLayoutMat[i])):
                returnStr += str(self.blockLayoutMat[i][j])
            returnStr += ">"
            returnStr += f" {self.blockSignsMat[i]} " if i < 7 else ""

        return returnStr
    
    @staticmethod
    def StateBlockDifference(_state1, _state2):
        differenceCount = 0
        #for i, block in enumerate(_state1.blockLayoutMat):
        for i in range(7):
            if _state1.blockLayoutMat[0][i] != _state2.blockLayoutMat[0][i]: differenceCount += 1
            #for j, value in enumerate(block):
                #if value != _state2.blockLayoutMat[i][j]: differenceCount += 1

        for i, value in enumerate(_state1.blockSignsMat):
            if value != _state2.blockSignsMat[i]: differenceCount += 1
        
        return differenceCount

    
    def encode(self):
        blockLayout = copy.deepcopy(self.blockLayoutMat)

        for i in range(len(blockLayout)):
            for j in range(len(blockLayout[i])):
                if blockLayout[i][j] == 0:
                    blockLayout[i][j] = zeroQubit
                elif blockLayout[i][j] == 1:
                    blockLayout[i][j] = oneQubit

        encodedBlocks = []
        for i in range(len(blockLayout)):
            encodedBlock = blockLayout[i][0]
            for j in range(1, len(blockLayout[i])):
                encodedBlock = np.kron(encodedBlock, blockLayout[i][j])
            encodedBlocks.append(encodedBlock)
        
        encodedState = encodedBlocks[0]
        for i in range(len(self.blockSignsMat)):
            if self.blockSignsMat[i] == "+":
                encodedState = encodedState + encodedBlocks[i+1]
            else:
                encodedState = encodedState - encodedBlocks[i+1]

        return encodedState
    
    def errorChannel(self, _p):
        blockLayout = copy.deepcopy(self.blockLayoutMat)
        blockSignsMat = copy.deepcopy(self.blockSignsMat)

        for i in range(7):
            if random.random() <= _p/2:
                #print(f"Bit flip at position {i+1}")
                for j in range(len(blockLayout)):
                    blockLayout[j][i] ^= 1
            elif random.random() <= _p/2:
                #print(f"Phase flip at position {i+1}")
                for j in range(len(blockLayout)):
                    if blockLayout[j][i] == 1:
                        blockSignsMat[j-1] = "-" if blockSignsMat[j-1] == "+" else "+"

        erroredState = SteaneState(blockLayout, blockSignsMat)
        return erroredState
    
    def errorCorrection(self):

        Z1Z3Z4Z5_eigenvalue = computeEigenvalue(Z1 @ Z3 @ Z4 @ Z5, self.statevector)
        Z1Z2Z3Z6_eigenvalue = computeEigenvalue(Z1 @ Z2 @ Z3 @ Z6, self.statevector)
        Z2Z3Z4Z7_eigenvalue = computeEigenvalue(Z2 @ Z3 @ Z4 @ Z7, self.statevector)

        # print(f"Z1Z3Z4Z5 eigenvalue: {Z1Z3Z4Z5_eigenvalue}")
        # print(f"Z1Z2Z3Z6 eigenvalue: {Z1Z2Z3Z6_eigenvalue}")
        # print(f"Z2Z3Z4Z7 eigenvalue: {Z2Z3Z4Z7_eigenvalue}")

        detectedBitFlipped = None
        if Z1Z3Z4Z5_eigenvalue == -1 and Z1Z2Z3Z6_eigenvalue == -1 and Z2Z3Z4Z7_eigenvalue == 1:
            detectedBitFlipped = 0
        elif Z1Z3Z4Z5_eigenvalue == 1 and Z1Z2Z3Z6_eigenvalue == -1 and Z2Z3Z4Z7_eigenvalue == -1:
            detectedBitFlipped = 1
        elif Z1Z3Z4Z5_eigenvalue == -1 and Z1Z2Z3Z6_eigenvalue == -1 and Z2Z3Z4Z7_eigenvalue == -1:
            detectedBitFlipped = 2
        elif Z1Z3Z4Z5_eigenvalue == -1 and Z1Z2Z3Z6_eigenvalue == 1 and Z2Z3Z4Z7_eigenvalue == -1:
            detectedBitFlipped = 3
        elif Z1Z3Z4Z5_eigenvalue == -1 and Z1Z2Z3Z6_eigenvalue == 1 and Z2Z3Z4Z7_eigenvalue == 1:
            detectedBitFlipped = 4
        elif Z1Z3Z4Z5_eigenvalue == 1 and Z1Z2Z3Z6_eigenvalue == -1 and Z2Z3Z4Z7_eigenvalue == 1:
            detectedBitFlipped = 5
        elif Z1Z3Z4Z5_eigenvalue == 1 and Z1Z2Z3Z6_eigenvalue == 1 and Z2Z3Z4Z7_eigenvalue == -1:
            detectedBitFlipped = 6

        if detectedBitFlipped != None:
            #print(f"Detected bit flip at position {detectedBitFlipped + 1}")
            for i in range(len(self.blockLayoutMat)):
                self.blockLayoutMat[i][detectedBitFlipped] ^= 1

        X1X3X4X5_eigenvalue = computeEigenvalue(X1 @ X3 @ X4 @ X5, self.statevector)
        X1X2X3X6_eigenvalue = computeEigenvalue(X1 @ X2 @ X3 @ X6, self.statevector)
        X2X3X4X7_eigenvalue = computeEigenvalue(X2 @ X3 @ X4 @ X7, self.statevector)

        # print(f"X1X3X4X5 eigenvalue: {X1X3X4X5_eigenvalue}")
        # print(f"X1X2X3X6 eigenvalue: {X1X2X3X6_eigenvalue}")
        # print(f"X2X3X4X7 eigenvalue: {X2X3X4X7_eigenvalue}")

        detectedPhaseFlipped = None
        if X1X3X4X5_eigenvalue == -1 and X1X2X3X6_eigenvalue == -1 and X2X3X4X7_eigenvalue == 1:
            detectedPhaseFlipped = 0
        elif X1X3X4X5_eigenvalue == 1 and X1X2X3X6_eigenvalue == -1 and X2X3X4X7_eigenvalue == -1:
            detectedPhaseFlipped = 1
        elif X1X3X4X5_eigenvalue == -1 and X1X2X3X6_eigenvalue == -1 and X2X3X4X7_eigenvalue == -1:
            detectedPhaseFlipped = 2
        elif X1X3X4X5_eigenvalue == -1 and X1X2X3X6_eigenvalue == 1 and X2X3X4X7_eigenvalue == -1:
            detectedPhaseFlipped = 3
        elif X1X3X4X5_eigenvalue == -1 and X1X2X3X6_eigenvalue == 1 and X2X3X4X7_eigenvalue == 1:
            detectedPhaseFlipped = 4
        elif X1X3X4X5_eigenvalue == 1 and X1X2X3X6_eigenvalue == -1 and X2X3X4X7_eigenvalue == 1:
            detectedPhaseFlipped = 5
        elif X1X3X4X5_eigenvalue == 1 and X1X2X3X6_eigenvalue == 1 and X2X3X4X7_eigenvalue == -1:
            detectedPhaseFlipped = 6

        if detectedPhaseFlipped != None:
            #print(f"Detected phase flip at position {detectedPhaseFlipped + 1}")
            for i in range(len(self.blockLayoutMat)):
                if self.blockLayoutMat[i][detectedPhaseFlipped] == 1:
                    self.blockSignsMat[i-1] = "+" if self.blockSignsMat[i-1] == "-" else "-"

        self.statevector = self.encode()
    
def computeEigenvalue(_gate, _stateVector):
    """
    Compute the eigenvalue associated with applying a gate to a state vector.

    Args:
    _gate (numpy.ndarray): The gate matrix.
    _state_vector (numpy.ndarray): The state vector.

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

blockLayoutMat = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 1]
]

blockSignsMat = [
            "+",
            "+",
            "+",
            "+",
            "+",
            "+",
            "+"
]

initTime = time.time()
probabilityList = np.arange(0.01, 0.25, 0.01)
errorList = []
for i in range(len(probabilityList)):
    totalErrors = 0
    for j in range(CYCLES):
        differenceCount = 0
        state = SteaneState(blockLayoutMat, blockSignsMat) # Working
        errorState = state.errorChannel(probabilityList[i])
        errorState.errorCorrection()                   
        if not np.all(state.statevector == errorState.statevector):
            #totalErrors += ShorState.StateBlockDifference(state, errorState)
            totalErrors += SteaneState.StateBlockDifference(state, errorState)
    errorList.append(totalErrors)
    print(f"Error probability: {probabilityList[i]}")
    print(f"Total errors after {CYCLES} cycles: {totalErrors}")
    print(f"Error rate: {totalErrors / (CYCLES)}")
    print("-----------------------------")

print(f"Time taken: {round(time.time() - initTime, 2)}s")

theoreticalModel = []
for i, prob in enumerate(probabilityList):
    theoreticalModel.append(1 - ((1-prob) ** 7) - 7 * prob * ((1 - prob) ** 6))

x = np.linspace(probabilityList[0], probabilityList[-1], 50)

z = np.polyfit(probabilityList, np.log(theoreticalModel), 3)
f = np.poly1d(z)
y = f(x)
 
plt.yscale("log")
 
theoreticalPlot = plt.plot(probabilityList, theoreticalModel, 'x',x, y, linestyle='dashed', color="black")

z = np.polyfit(probabilityList, [i / ( CYCLES) for i in errorList], 3)
f = np.poly1d(z)
y = f(x)

dataPlot = plt.plot(probabilityList, [i / (CYCLES) for i in errorList], 'o', x, y, color="blue")

plt.ylim(10**-3, 1)
plt.show()
