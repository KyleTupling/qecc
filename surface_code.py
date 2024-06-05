import time
import random
import copy
import numpy as np 
import matplotlib.pyplot as plt
import csv

import matplotlib 

font = {'family' : 'serif',
         'size'   : 12,
         'serif':  'cmr10'
         }

matplotlib.rc('font', **font)

CYCLES = 1000

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1], [1, 0]])
I = np.eye(2)

# Scales an operator for an n-qubit system
def scaleOperator(_operator, _n, _pos):
    """Scales an operator for an n-qubit system

    Args:
        _operator (array): operator matrix for a single-qubit system
        _n (int): number of qubits to scale for
        _pos (int): which qubit the operator acts on in the n-qubit system
    
    Returns:
        array: scaled operator matrix

    """
    if _pos == 1:
        newOperator = _operator
        for i in range(_n-1):
            newOperator = np.kron(newOperator, I)
    else:
        newOperator = I
        for i in range(_pos - 2):
            newOperator = np.kron(newOperator, I)
        newOperator = np.kron(newOperator, _operator)
        for i in range(_n - _pos):
            newOperator = np.kron(newOperator, I)
    return newOperator

class Qubit(object):

    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

        self.data = 0
        self.positive = True
        self.vector = np.matrix([[1], [0]])

        self.surfaceNumber = None
        self.localXStabilisers = []
        self.localZStabilisers = []

    @staticmethod
    def EqualStatevectors(_qubit1, _qubit2):
        return _qubit1.vector[0] == _qubit2.vector[0] and _qubit1.vector[1] == _qubit2.vector[1]

    def setValue(self, _value):
        if _value != 0 and _value != 1:
            raise ValueError("Qubit must have data value of 0 or 1!")
        
        self.data = _value
        if _value == 0:
            self.vector = np.matrix([[1], [0]])
        elif _value == 1:
            self.vector = np.matrix([[0], [1]])

    def flipSign(self):
        self.positive = not self.positive
        self.vector *= -1

    def bitFlip(self):
        self.data ^= 1
        self.vector = X @ self.vector

    def phaseFlip(self):
        if self.data != 0:
            #self.data == -self.data
            self.positive = not self.positive
            self.vector = Z @ self.vector

    def bitAndPhaseFlip(self):
        if self.data == 0:
            self.data = 1
        elif abs(self.data) == 1:
            self.data = 0
            self.positive = not self.positive
        self.vector = Y @ self.vector
    
class QubitSurface(object):

    def __init__(self, _surfaceLayout):
        self.surfaceLayout = _surfaceLayout

        self.qubitAmount = 0
        self.qubitReferences = []

        self.surface = []

        # Populate surface with data qubits
        for i in range(len(self.surfaceLayout)):
            surfaceRow = []
            for j in range(len(self.surfaceLayout[i])):
                if self.surfaceLayout[i][j] == "q":
                    self.qubitAmount += 1
                    qubit = Qubit(i, j)
                    qubit.surfaceNumber = self.qubitAmount
                    qubit.data = 0
                    surfaceRow.append(qubit)
                    self.qubitReferences.append(qubit)
                else: # Insert stabiliser qubit representations
                    surfaceRow.append(self.surfaceLayout[i][j])
            self.surface.append(surfaceRow)

        self.xStabilisers = []
        self.zStabilisers = []

        # TODO: Can simplify the k loop by checking stabiliser type before appending to stabiliser list
        for i in range(len(self.surface)):
            for j in range(len(self.surface[i])):
                if self.surface[i][j] == "x":
                    neighbours = self.getNeighbours(i, j)
                    stabiliserNumberList = []
                    for k in range(len(neighbours)):
                        if isinstance(neighbours[k], Qubit):
                            # Craft appropriate stabilisers and insert into grid positions ready for measurements
                            stabiliserNumberList.append(neighbours[k].surfaceNumber)
                    self.xStabilisers.append(stabiliserNumberList)
                elif self.surface[i][j] == "z":
                    neighbours = self.getNeighbours(i, j)
                    stabiliserNumberList = []
                    for k in range(len(neighbours)):
                        if isinstance(neighbours[k], Qubit):
                            # Craft appropriate stabilisers and insert into grid positions ready for measurements
                            stabiliserNumberList.append(neighbours[k].surfaceNumber)
                    self.zStabilisers.append(stabiliserNumberList)
    
    # This comparison function will only work if the grid was initialised with the same value for every data qubit
    def surfaceQubitErrorCount(self, _globalInitialValue):
        totalErrors = 0
        for i in range(len(self.qubitReferences)):
            if self.qubitReferences[i].data != _globalInitialValue:
                totalErrors += 1
            
            if not self.qubitReferences[i].positive:
                totalErrors += 1

        return totalErrors
    
    # Returns the majority check qubit data and sign
    def qubitMajorityCheck(self):
        majorityData = 0
        majorityPositive = True

        totalZeros = 0
        totalPositive = 0

        for i in range(len(self.qubitReferences)):
            if self.qubitReferences[i].data == 0:
                totalZeros += 1
            if self.qubitReferences[i].positive:
                totalPositive += 1

        if totalZeros < len(self.qubitReferences) / 2:
            majorityData = 1
        if totalPositive < len(self.qubitReferences) / 2:
            majorityPositive = False

        return majorityData, majorityPositive

    def errorChannel(self, _p):
        for i in range(len(self.qubitReferences)):
            if random.random() <= _p/3: # Bit-flip
                self.qubitReferences[i].bitFlip()
            elif random.random() <= _p/3:   # Phase-flip
                self.qubitReferences[i].phaseFlip()
            # TODO: Fix B+P errors
            elif random.random() <= _p/3: # Bit and phase-flip
                self.qubitReferences[i].bitAndPhaseFlip()

    # TODO: Collect syndrome table for bit-flips errors and phase-flip errors
    def getErrorSyndromes(self):
        zSyndrome = []
        for stabiliserList in self.zStabilisers:
            eigenvalue = 1
            for stabiliserNumber in stabiliserList:
                if self.qubitReferences[stabiliserNumber - 1].data == 1:
                    eigenvalue *= -1
            zSyndrome.append(0 if eigenvalue == 1 else 1)

        xSyndrome = []
        for stabiliserList in self.xStabilisers:
            eigenvalue = 1
            for stabiliserNumber in stabiliserList:
                # if self.qubitReferences[stabiliserNumber - 1].data == -1:
                if not self.qubitReferences[stabiliserNumber - 1].positive:
                    eigenvalue *= -1
            xSyndrome.append(0 if eigenvalue == 1 else 1)

        # print(f"Z Syndrome: {zSyndrome}")
        # print(f"X Syndrome: {xSyndrome}")

        return xSyndrome, zSyndrome

    # TODO: Make this scale with any size surface (find relation between syndrome vector and position on grid?)
    def errorCorrection(self, _xSyndrome, _zSyndrome):

        # Correct single phase-flip errors
        if _xSyndrome == [1, 0, 0, 0, 0, 0]:
            self.qubitReferences[0].flipSign()
        elif _xSyndrome == [0, 1, 0, 0, 0, 0]:
            self.qubitReferences[1].flipSign()
        elif _xSyndrome == [0, 0, 1, 0, 0, 0]:
            self.qubitReferences[2].flipSign()
        elif _xSyndrome == [1, 1, 0, 0, 0, 0]:
            self.qubitReferences[3].flipSign()
        elif _xSyndrome == [0, 1, 1, 0, 0, 0]:
            self.qubitReferences[4].flipSign()
        elif _xSyndrome == [1, 0, 0, 1, 0, 0]:
            self.qubitReferences[5].flipSign()
        elif _xSyndrome == [0, 1, 0, 0, 1, 0]:
            self.qubitReferences[6].flipSign()
        elif _xSyndrome == [0, 0, 1, 0, 0, 1]:
            self.qubitReferences[7].flipSign()
        elif _xSyndrome == [0, 0, 0, 1, 1, 0]:
            self.qubitReferences[8].flipSign()
        elif _xSyndrome == [0, 0, 0, 0, 1, 1]:
            self.qubitReferences[9].flipSign()
        elif _xSyndrome == [0, 0, 0, 1, 0, 0]:
            self.qubitReferences[10].flipSign()
        elif _xSyndrome == [0, 0, 0, 0, 1, 0]:
            self.qubitReferences[11].flipSign()
        elif _xSyndrome == [0, 0, 0, 0, 0, 1]:
            self.qubitReferences[12].flipSign()

        # Correct single bit-flip errors
        if _zSyndrome == [1, 0, 0, 0, 0, 0]:
            self.qubitReferences[0].bitFlip()
        elif _zSyndrome == [0, 1, 0, 0, 0, 0]:
            self.qubitReferences[1].bitFlip()
        elif _zSyndrome == [0, 0, 1, 0, 0, 0]:
            self.qubitReferences[2].bitFlip()
        elif _zSyndrome == [1, 1, 0, 0, 0, 0]:
            self.qubitReferences[3].bitFlip()
        elif _zSyndrome == [0, 1, 1, 0, 0, 0]:
            self.qubitReferences[4].bitFlip()
        elif _zSyndrome == [1, 0, 0, 1, 0, 0]:
            self.qubitReferences[5].bitFlip()
        elif _zSyndrome == [0, 1, 0, 0, 1, 0]:
            self.qubitReferences[6].bitFlip()
        elif _zSyndrome == [0, 0, 1, 0, 0, 1]:
            self.qubitReferences[7].bitFlip()
        elif _zSyndrome == [0, 0, 0, 1, 1, 0]:
            self.qubitReferences[8].bitFlip()
        elif _zSyndrome == [0, 0, 0, 0, 1, 1]:
            self.qubitReferences[9].bitFlip()
        elif _zSyndrome == [0, 0, 0, 1, 0, 0]:
            self.qubitReferences[10].bitFlip()
        elif _zSyndrome == [0, 0, 0, 0, 1, 0]:
            self.qubitReferences[11].bitFlip()
        elif _zSyndrome == [0, 0, 0, 0, 0, 1]:
            self.qubitReferences[12].bitFlip()

    def getSurfaceCoordinates(self, _strType, _occurNum):
        occurrenceCount = 0
        for i in range(len(self.surfaceLayout)):
            for j in range(len(self.surfaceLayout[i])):
                if self.surfaceLayout[i][j] == _strType:
                    occurrenceCount += 1
                    if occurrenceCount == _occurNum:
                        return i, j

    def getNeighbours(self, _rowPos, _colPos):
        neighbours = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                # Exclude element self-check
                if i == 0 and j == 0:
                    continue
                rowNeighbour = _rowPos + i
                columnNeighbour = _colPos + j
                if 0 <= rowNeighbour < len(self.surface) and 0 <= columnNeighbour < len(self.surface[i]):
                    neighbours.append(self.surface[rowNeighbour][columnNeighbour])
        
        return neighbours
                    
    def __str__(self):
        returnStr = ""
        for i in range(len(self.surfaceLayout)):
            for j in range(len(self.surfaceLayout[i])):
                returnStr += f"{self.surfaceLayout[i][j]} "
            returnStr += "\n"
        return returnStr
    
    def qubitValuesStr(self):
        returnStr = ""
        for qubit in self.qubitReferences:
            returnStr += "+" if qubit.positive else "-"
            returnStr += f"{qubit.data} "
        return returnStr

gridLayout = [
    ["q", "x", "q", "x", "q"],
    ["z", "q", "z", "q", "z"],
    ["q", "x", "q", "x", "q"],
    ["z", "q", "z", "q", "z"],
    ["q", "x", "q", "x", "q"]
]

surface = QubitSurface(gridLayout)
surface.qubitReferences[0].bitAndPhaseFlip()
surface.qubitReferences[0].phaseFlip()
print([qubit.data for qubit in surface.qubitReferences])
print([qubit.positive for qubit in surface.qubitReferences])
xSyndrome, zSyndrome = surface.getErrorSyndromes()
print(f"X Syndrome: {xSyndrome}")
print(f"Z Syndrome: {zSyndrome}")
surface.errorCorrection(xSyndrome, zSyndrome)
print([qubit.positive for qubit in surface.qubitReferences])
print([qubit.data for qubit in surface.qubitReferences])

initTime = time.time()
probabilityList = np.arange(0.005, 0.25, 0.005)
errorList = []
for i in range(len(probabilityList)):
    totalErrors = 0
    for j in range(CYCLES):
        surface = QubitSurface(gridLayout)
        initialQubitData = [copy.deepcopy(qubit) for qubit in surface.qubitReferences]
        initialQubitSigns = [qubit.positive for qubit in surface.qubitReferences]
        surface.errorChannel(probabilityList[i])
        xSyndrome, zSyndrome = surface.getErrorSyndromes()
        surface.errorCorrection(xSyndrome, zSyndrome)
        finalQubitData = [qubit for qubit in surface.qubitReferences]
        finalQubitSigns = [qubit.positive for qubit in surface.qubitReferences]
        if surface.surfaceQubitErrorCount(0) >= 2:
            totalErrors += 1
    errorList.append(totalErrors)
    print(f"Error probability: {probabilityList[i]}")
    print(f"Total errors after {CYCLES} cycles: {totalErrors}")
    print(f"Error rate: {totalErrors / (CYCLES)}")
    print("-----------------------------")

print(f"Time taken: {round(time.time() - initTime, 2)}s")

with open('data/surface_code.csv', 'w') as dataFile:
    writer = csv.writer(dataFile, delimiter=",")
    writer.writerow([error / CYCLES for error in errorList])

theoreticalModel = []
for i, prob in enumerate(probabilityList):
    theoreticalModel.append(1 - ((1-prob) ** 13) - 13 * prob * ((1 - prob) ** 12))

x = np.linspace(probabilityList[0], probabilityList[-1], 100)
theoreticalX = np.linspace(0, probabilityList[-1], 100)

z = np.polyfit(probabilityList, np.log(theoreticalModel), 3)
f = np.poly1d(z)
y = 1 - ((1-theoreticalX) ** 13) - 13 * theoreticalX * ((1 - theoreticalX) ** 12)

theoreticalPlot = plt.plot(theoreticalX, y, linestyle='dashed', color="black", label="Theoretical model")

z = np.polyfit(probabilityList, [i / (CYCLES) for i in errorList], 3)
f = np.poly1d(z)
y = f(x)

plt.yscale("log")
dataPlot = plt.plot(probabilityList, [i / (CYCLES) for i in errorList], 'x', color="blue") 
dataCurvePlot = plt.plot(x, y, color="blue", label="Simulation")

#plt.title(f"Surface Depolarising Probability vs. QBER @{CYCLES} cycles")
plt.xlabel(r"Depolarising probability (p)", fontsize=16)
plt.ylabel(r"QBER", fontsize=16)
plt.legend(loc="lower right", fontsize=14)

# Limit y-axis
plt.ylim(10**-4, 10^0)

plt.plot()

plt.autoscale()

plt.show()
