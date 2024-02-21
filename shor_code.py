import random
import numpy as np 
import copy
import time
np.set_printoptions(threshold=np.inf) # Allow for printing of full matrices etc

DEBUG_MODE = False
CYCLES = 5000

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
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
Z1 = np.kron(Z, kronScaleOperator(I, 8))
Z2 = np.kron(I, np.kron(Z, kronScaleOperator(I, 7)))
Z3 = np.kron(kronScaleOperator(I, 2), np.kron(Z, kronScaleOperator(I, 6)))
Z4 = np.kron(kronScaleOperator(I, 3), np.kron(Z, kronScaleOperator(I, 5)))
Z5 = np.kron(kronScaleOperator(I, 4), np.kron(Z, kronScaleOperator(I, 4)))
Z6 = np.kron(kronScaleOperator(I, 5), np.kron(Z, kronScaleOperator(I, 3)))
Z7 = np.kron(kronScaleOperator(I, 6), np.kron(Z, kronScaleOperator(I, 2)))
Z8 = np.kron(kronScaleOperator(I, 7), np.kron(Z, I))
Z9 = np.kron(kronScaleOperator(I, 8), Z)
ZOperators = [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9]

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
XOperators = [X1, X2, X3, X4, X5, X6, X7, X8, X9]

class ShorState(object):

    def __init__(self, _blockLayoutMat, _blockSignsMat):
        self.blockLayoutMat = _blockLayoutMat
        self.blockSignsMat = _blockSignsMat

        # Keep track of errors induced
        self.bitFlipCount = 0
        self.phaseFlipCount = 0

        # Encode
        self.statevector = self.encode()

    # Format printing to be human-readable
    def __str__(self):
        returnStr = ""
        for i, block in enumerate(self.blockLayoutMat):
            returnStr += "("
            for j, subblock in enumerate(block):
                returnStr += "|"
                for k, value in enumerate(subblock):
                    returnStr += str(value)
                returnStr += ">"
                returnStr += f" {self.blockSignsMat[i][0]} " if j % 2 == 0 else ""
            returnStr += ") x " if i < 2 else ") "
                    
        return returnStr

    def encode(self):
        blockLayout = copy.deepcopy(self.blockLayoutMat)

        for i, block in enumerate(blockLayout):
            for j, subblock in enumerate(block):
                for k, value in enumerate(subblock):
                    if value == 0:
                        blockLayout[i][j][k] = zeroQubit
                    elif value == 1:
                        blockLayout[i][j][k] = oneQubit
                    elif value == -1:
                        blockLayout[i][j][k] = negOneQubit

        firstBlockLayout = blockLayout[0]
        secondBlockLayout = blockLayout[1]
        thirdBlockLayout = blockLayout[2]

        blockSignsMat = copy.deepcopy(self.blockSignsMat)

        # Assign signs to encoding
        if blockSignsMat[0][0] == "+":
            firstBlock = np.kron(np.kron(firstBlockLayout[0][0], firstBlockLayout[0][1]), firstBlockLayout[0][2]) + np.kron(np.kron(firstBlockLayout[1][0], firstBlockLayout[1][1]), firstBlockLayout[1][2])
        else:
            firstBlock = np.kron(np.kron(firstBlockLayout[0][0], firstBlockLayout[0][1]), firstBlockLayout[0][2]) - np.kron(np.kron(firstBlockLayout[1][0], firstBlockLayout[1][1]), firstBlockLayout[1][2])
        
        if blockSignsMat[1][0] == "+":
            secondBlock = np.kron(np.kron(secondBlockLayout[0][0], secondBlockLayout[0][1]), secondBlockLayout[0][2]) + np.kron(np.kron(secondBlockLayout[1][0], secondBlockLayout[1][1]), secondBlockLayout[1][2])
        else:
            secondBlock = np.kron(np.kron(secondBlockLayout[0][0], secondBlockLayout[0][1]), secondBlockLayout[0][2]) - np.kron(np.kron(secondBlockLayout[1][0], secondBlockLayout[1][1]), secondBlockLayout[1][2])
        
        if blockSignsMat[2][0] == "+":
            thirdBlock = np.kron(np.kron(thirdBlockLayout[0][0], thirdBlockLayout[0][1]), thirdBlockLayout[0][2]) + np.kron(np.kron(thirdBlockLayout[1][0], thirdBlockLayout[1][1]), thirdBlockLayout[1][2])
        else:
            thirdBlock = np.kron(np.kron(thirdBlockLayout[0][0], thirdBlockLayout[0][1]), thirdBlockLayout[0][2]) - np.kron(np.kron(thirdBlockLayout[1][0], thirdBlockLayout[1][1]), thirdBlockLayout[1][2])

        return np.kron(firstBlock, np.kron(secondBlock, thirdBlock))
    
    # REFACTOR: NEEDS TO BE EITHER BIT FLIP, PHASE FLIP, BOTH, OR NONE
    def errorChannel(self, p):
        blockLayout = copy.deepcopy(self.blockLayoutMat)
        blockSignsMat = copy.deepcopy(self.blockSignsMat)

        if DEBUG_MODE: print("-- ERROR CHANNEL --------------------")

        #Bit-flipper
        for i, block in enumerate(blockLayout):
            for j in range(3):
                if random.random() <= p:
                    self.bitFlipCount += 1 # Increase instance bit flip count by 1
                    if DEBUG_MODE: print(f"Bits at block {i+1}, index {j+1} flipping")
                    block[0][j] ^= 1
                    block[1][j] ^= 1
                elif random.random() <= p:
                    self.phaseFlipCount += 1
                    if DEBUG_MODE: print(f"Phase flipping at block {i + 1}")
                    if block[1][j] == 1:
                        blockSignsMat[i][0] = "-" if blockSignsMat[i][0] == "+" else "+"
                    

        # for i, block in enumerate(blockSignsMat):
        #     if random.random() <= p:
        #         self.phaseFlipCount += 1 # Increase instance phase flip count by 1
        #         print(f"Phase flipping at block {i + 1}")
        #         if block[0] == "+":
        #             block[0] = "-"
        #         else:
        #             block[0] == "+"
        if DEBUG_MODE: print("-------------------------------------")
        
        return ShorState(blockLayout, blockSignsMat)

    def errorCorrection(self):
        sv = self.statevector

        Z1Z2_eigenvalue = computeEigenvalue(Z1 @ Z2, sv)
        Z2Z3_eigenvalue = computeEigenvalue(Z2 @ Z3, sv)
        Z4Z5_eigenvalue = computeEigenvalue(Z4 @ Z5, sv)
        Z5Z6_eigenvalue = computeEigenvalue(Z5 @ Z6, sv)
        Z7Z8_eigenvalue = computeEigenvalue(Z7 @ Z8, sv)
        Z8Z9_eigenvalue = computeEigenvalue(Z8 @ Z9, sv)

        if Z1Z2_eigenvalue == -1 and Z2Z3_eigenvalue == 1:
            #print("Bit flip detected on 1st block bit 1")
            self.blockLayoutMat[0][0][0] ^= 1
            self.blockLayoutMat[0][1][0] ^= 1
        elif Z1Z2_eigenvalue == -1 and Z2Z3_eigenvalue == -1:
            #print("Bit flip detected on 1st block bit 2")
            self.blockLayoutMat[0][0][1] ^= 1
            self.blockLayoutMat[0][1][1] ^= 1
        elif Z1Z2_eigenvalue == 1 and Z2Z3_eigenvalue == -1:
            #print("Bit flip detected on 1st block bit 3")
            self.blockLayoutMat[0][0][2] ^= 1
            self.blockLayoutMat[0][1][2] ^= 1

        if Z4Z5_eigenvalue == -1 and Z5Z6_eigenvalue == 1:
            #print("Bit flip detected on 2nd block bit 1")
            self.blockLayoutMat[1][0][0] ^= 1
            self.blockLayoutMat[1][1][0] ^= 1
        elif Z4Z5_eigenvalue == -1 and Z5Z6_eigenvalue == -1:
            #print("Bit flip detected on 2nd block bit 2")
            self.blockLayoutMat[1][0][1] ^= 1
            self.blockLayoutMat[1][1][1] ^= 1
        elif Z4Z5_eigenvalue == 1 and Z5Z6_eigenvalue == -1:
            #print("Bit flip detected on 2nd block bit 3")
            self.blockLayoutMat[1][0][2] ^= 1
            self.blockLayoutMat[1][1][2] ^= 1

        if Z7Z8_eigenvalue == -1 and Z8Z9_eigenvalue == 1:
            #print("Bit flip detected on 3rd block bit 1")
            self.blockLayoutMat[2][0][0] ^= 1
            self.blockLayoutMat[2][1][0] ^= 1
        elif Z7Z8_eigenvalue == -1 and Z8Z9_eigenvalue == -1:
            #print("Bit flip detected on 3rd block bit 2")
            self.blockLayoutMat[2][0][1] ^= 1
            self.blockLayoutMat[2][1][1] ^= 1
        elif Z7Z8_eigenvalue == 1 and Z8Z9_eigenvalue == -1:
            #print("Bit flip detected on 3rd block bit 3")
            self.blockLayoutMat[2][0][2] ^= 1
            self.blockLayoutMat[2][1][2] ^= 1

        PHASE1_eigenvalue = computeEigenvalue(X1 @ X2 @ X3 @ X4 @ X5 @ X6, sv)
        PHASE2_eigenvalue = computeEigenvalue(X4 @ X5 @ X6 @ X7 @ X8 @ X9, sv)

        if PHASE1_eigenvalue == -1 and PHASE2_eigenvalue == 1:
            self.blockSignsMat[0][0] = "+" if self.blockSignsMat[0][0] == "-" else "-"
        elif PHASE1_eigenvalue == -1 and PHASE2_eigenvalue == -1:
            self.blockSignsMat[1][0] = "+" if self.blockSignsMat[1][0] == "-" else "-"
        elif PHASE1_eigenvalue == 1 and PHASE2_eigenvalue == -1:
            self.blockSignsMat[2][0] = "+" if self.blockSignsMat[2][0] == "-" else "-"
        
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
    magnitude = np.abs(innerProduct)
    
    # Compute the eigenvalue
    eigenvalue = innerProduct / magnitude
    
    return eigenvalue

# MAY NOT BE NEEDED
def baseBlockDifference(_block1, _block2):
    if len(_block1) != len(_block2):
        raise ValueError("Blocks must be the same length")
    
    differenceCount = 0
    for i, block in enumerate(_block1):
            for j, subblock in enumerate(block):
                for k, value in enumerate(subblock):
                    if value != _block2[i][j][k]:
                        differenceCount += 1

    return differenceCount

# NEED TO ADD FUNCTIONALITY FOR PHASE FLIP NOTATION WITHIN BLOCK VIEW
baseBlock = [
    [[0, 0, 0], [1, 1, 1]], 
    [[0, 0, 0], [1, 1, 1]], 
    [[0, 0, 0], [1, 1, 1]]
]

# Use signs matrix?
# - Could use this to allow for phase flipping and for encoding the |1> state
baseBlockSigns = [
    ["+"],
    ["+"],
    ["+"]
]

#sv = shorStatevector(baseBlock)

state = ShorState(baseBlock, baseBlockSigns) # Working
errorState = state.errorChannel(0.2)
sv = errorState.statevector
print(f"Initial state: {state}")
print(f"Error state: {errorState}")
errorState.errorCorrection()
print(f"Corrected state: {errorState}")
if DEBUG_MODE:
    print("-- STATES ---------------------------")
    print(f"Initial state: {state}")
    print(f"Errored state: {errorState}")
    print("-------------------------------------")

# No Qubit Flip = (+1, +1)
# 1st Qubit Flip = (-1, +1)
# 2nd Qubit Flip = (-1, -1)
# 3rd Qubit Flip = (+1, -1)

Z1Z2_eigenvalue = computeEigenvalue(Z1 @ Z2, sv)
Z2Z3_eigenvalue = computeEigenvalue(Z2 @ Z3, sv)
Z4Z5_eigenvalue = computeEigenvalue(Z4 @ Z5, sv)
Z5Z6_eigenvalue = computeEigenvalue(Z5 @ Z6, sv)
Z7Z8_eigenvalue = computeEigenvalue(Z7 @ Z8, sv)
Z8Z9_eigenvalue = computeEigenvalue(Z8 @ Z9, sv)

if DEBUG_MODE:
    # Bit flip error detection
    print(f"Z1 Z2 |psi> eigenvalue: {Z1Z2_eigenvalue}")
    print(f"Z2 Z3 |psi> eigenvalue: {Z2Z3_eigenvalue}")
    print("-------------------------------")
    print(f"Z4 Z5 |psi> eigenvalue: {Z4Z5_eigenvalue}")
    print(f"Z5 Z6 |psi> eigenvalue: {Z5Z6_eigenvalue}")
    print("-------------------------------")
    print(f"Z7 Z8 |psi> eigenvalue: {Z7Z8_eigenvalue}")
    print(f"Z8 Z9 |psi> eigenvalue: {Z8Z9_eigenvalue}")
    print("-------------------------------")
    print(f"X1 X2 X3 X4 X5 X6 |psi> eigenvalue: {computeEigenvalue(X1 @ X2 @ X3 @ X4 @ X5 @ X6, sv)}")
    print(f"X4 X5 X6 X7 X8 X9 |psi> eigenvalue: {computeEigenvalue(X4 @ X5 @ X6 @ X7 @ X8 @ X9, sv)}")
    print("-------------------------------")

if DEBUG_MODE:
    # Detect bitflips
    if Z1Z2_eigenvalue == -1 and Z2Z3_eigenvalue == 1:
        print("Bit flip detected on 1st block bit 1")
    elif Z1Z2_eigenvalue == -1 and Z2Z3_eigenvalue == -1:
        print("Bit flip detected on 1st block bit 2")
    elif Z1Z2_eigenvalue == 1 and Z2Z3_eigenvalue == -1:
        print("Bit flip detected on 1st block bit 3")

    if Z4Z5_eigenvalue == -1 and Z5Z6_eigenvalue == 1:
        print("Bit flip detected on 2nd block bit 1")
    elif Z4Z5_eigenvalue == -1 and Z5Z6_eigenvalue == -1:
        print("Bit flip detected on 2nd block bit 2")
    elif Z4Z5_eigenvalue == 1 and Z5Z6_eigenvalue == -1:
        print("Bit flip detected on 2nd block bit 3")

    if Z7Z8_eigenvalue == -1 and Z8Z9_eigenvalue == 1:
        print("Bit flip detected on 3rd block bit 1")
    elif Z7Z8_eigenvalue == -1 and Z8Z9_eigenvalue == -1:
        print("Bit flip detected on 3rd block bit 2")
    elif Z7Z8_eigenvalue == 1 and Z8Z9_eigenvalue == -1:
        print("Bit flip detected on 3rd block bit 3")

initTime = time.time()
totalErrors = 0
for i in range(CYCLES):
    differenceCount = 0
    state = ShorState(baseBlock, baseBlockSigns) # Working
    errorState = state.errorChannel(0.2)
    errorState.errorCorrection()
    if not np.all(state.statevector == errorState.statevector):
        totalErrors += 1

print(f"Total errors after {CYCLES} cycles: {totalErrors}")
print(f"Error rate: {totalErrors / (CYCLES)}")
print(f"Time taken: {time.time() - initTime}")
