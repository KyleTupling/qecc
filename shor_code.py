import numpy as np 
np.set_printoptions(threshold=np.inf) # Allow for printing of full matrices etc

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# Scale operator to n qubit system
def kronScaleOperator(_operator, _n):
    newOperator = _operator
    for i in range(_n-1):
        newOperator = np.kron(newOperator, I)
    return newOperator

zeroQubit = np.matrix([[1], [0]])
oneQubit = np.matrix([[0], [1]])
negOneQubit = np.matrix([[0], [-1]])


def computeEigenvalue(_gate, _state_vector):
    """
    Compute the eigenvalue associated with applying a gate to a state vector.

    Args:
    _gate (numpy.ndarray): The gate matrix.
    _state_vector (numpy.ndarray): The state vector.

    Returns:
    complex: The eigenvalue associated with the gate.
    """
    # Apply the gate to the state vector
    result_vector = np.dot(_gate, _state_vector)
    
    # Compute the inner product of the resulting state vector with the original state vector
    inner_product = np.dot(_state_vector.conj().T, result_vector)
    
    # Compute the magnitude of the inner product
    magnitude = np.abs(inner_product)
    
    # Compute the eigenvalue
    eigenvalue = inner_product / magnitude
    
    return eigenvalue

# ONLY WORKS FOR 0 QUBIT
def shorStatevector(_blockLayoutMat):
    """
    Encode a given qubit using the Shor code

    Args:
    _blockLayoutMat (array): An array giving the numerical layout of the blocks in the code.

    Returns:
    numpy.ndarray: The encoded statevector
    """
    blockLayoutMat = _blockLayoutMat
    for i, block in enumerate(_blockLayoutMat):
        for j, subblock in enumerate(block):
            for k, value in enumerate(subblock):
                if value == 0:
                    blockLayoutMat[i][j][k] = zeroQubit
                elif value == 1:
                    blockLayoutMat[i][j][k] = oneQubit
                elif value == -1:
                    blockLayoutMat[i][j][k] = negOneQubit

    firstBlockLayout = blockLayoutMat[0]
    secondBlockLayout = blockLayoutMat[1]
    thirdBlockLayout = blockLayoutMat[2]

    firstBlock = np.kron(np.kron(firstBlockLayout[0][0], firstBlockLayout[0][1]), firstBlockLayout[0][2]) + np.kron(np.kron(firstBlockLayout[1][0], firstBlockLayout[1][1]), firstBlockLayout[1][2])
    secondBlock = np.kron(np.kron(secondBlockLayout[0][0], secondBlockLayout[0][1]), secondBlockLayout[0][2]) + np.kron(np.kron(secondBlockLayout[1][0], secondBlockLayout[1][1]), secondBlockLayout[1][2])
    thirdBlock = np.kron(np.kron(thirdBlockLayout[0][0], thirdBlockLayout[0][1]), thirdBlockLayout[0][2]) + np.kron(np.kron(thirdBlockLayout[1][0], thirdBlockLayout[1][1]), thirdBlockLayout[1][2])

    # firstBlock = np.kron(firstBlockLayout[0][0], np.kron(firstBlockLayout[0][1], firstBlockLayout[0][2])) + np.kron(firstBlockLayout[1][0], np.kron(firstBlockLayout[1][1], firstBlockLayout[1][2]))
    # secondBlock = np.kron(secondBlockLayout[0][0], np.kron(secondBlockLayout[0][1], secondBlockLayout[0][2])) + np.kron(secondBlockLayout[1][0], np.kron(secondBlockLayout[1][1], secondBlockLayout[1][2]))
    # thirdBlock = np.kron(thirdBlockLayout[0][0], np.kron(thirdBlockLayout[0][1], thirdBlockLayout[0][2])) + np.kron(thirdBlockLayout[1][0], np.kron(thirdBlockLayout[1][1], thirdBlockLayout[1][2]))

    return np.kron(firstBlock, np.kron(secondBlock, thirdBlock))


# NEED TO ADD FUNCTIONALITY FOR PHASE FLIP NOTATION WITHIN BLOCK VIEW
baseBlock = [
    [[1, 0, 0], [0, 1, 1]], 
    [[0, 0, 0], [1, 1, 1]], 
    [[0, 0, 0], [1, 1, 1]]
    ]
sv = shorStatevector(baseBlock)

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

# No Qubit Flip = (+1, +1)
# 1st Qubit Flip = (-1, +1)
# 2nd Qubit Flip = (-1, -1)
# 3rd Qubit Flip = (+1, -1)

# Bit flip error detection
print(f"Z1 Z2 |psi> eigenvalue: {np.matrix.sum(Z1 @ Z2 @ sv)/np.matrix.sum(sv)}")
print(f"Z2 Z3 |psi> eigenvalue: {np.matrix.sum(Z2 @ Z3 @ sv)/np.matrix.sum(sv)}")
print("-------------------------------")
print(f"Z4 Z5 |psi> eigenvalue: {np.matrix.sum(Z4 @ Z5 @ sv)/np.matrix.sum(sv)}")
print(f"Z5 Z6 |psi> eigenvalue: {np.matrix.sum(Z5 @ Z6 @ sv)/np.matrix.sum(sv)}")
print("-------------------------------")
print(f"Z7 Z8 |psi> eigenvalue: {np.matrix.sum(Z7 @ Z8 @ sv)/np.matrix.sum(sv)}")
print(f"Z8 Z9 |psi> eigenvalue: {np.matrix.sum(Z8 @ Z9 @ sv)/np.matrix.sum(sv)}")
print("-------------------------------")
print(f"X1 X2 X3 X4 X5 X6 |psi> eigenvalue: {np.matrix.sum(X1 @ X2 @ X3 * X4 @ X5 @ X6 @ sv)/np.matrix.sum(sv)}")
print(f"X4 X5 X6 X7 X8 X9 |psi> eigenvalue: {np.matrix.sum(X4 * X5 * X6 * X7 * X8 * X9 * sv)/np.matrix.sum(sv)}")


print(computeEigenvalue(Z1 @ Z2, sv))
print(computeEigenvalue(Z2 @ Z3, sv))

# -|111> isn't the same as |-1 -1 -1 > !!!