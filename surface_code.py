import time
import random
import numpy as np 
import matplotlib.pyplot as plt

CYCLES = 1000

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
I = np.eye(2)

def scaleOperator(_operator, _n, _pos):
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
        self.vector = np.matrix([[1], [0]])

        self.surfaceNumber = None
        self.localXStabilisers = []
        self.localZStabilisers = []

    def setValue(self, _value):
        if _value != 0 and _value != 1:
            raise ValueError("Qubit must have data value of 0 or 1!")
        
        self.data = _value
        if _value == 0:
            self.vector = np.matrix([[1], [0]])
        elif _value == 1:
            self.vector = np.matrix([[0], [1]])

    def bitFlip(self):
        self.data ^= 1

# Takes a pattern array
# Currently only modelled to handle n, n - 1, n, n -1, etc. where n is odd > 1
class QubitSurface(object):

    def __init__(self, _patternArray):
        self.grid = []
        self.qubitCount = 0

        for x in range(len(_patternArray)):
            row = []
            for y in range(_patternArray[x]):
                qubit = Qubit(x, y)
                row.append(qubit)
                self.qubitCount += 1
                qubit.surfaceNumber = self.qubitCount
            self.grid.append(row)

        self.xStabilisers = []
        for i in range(self.qubitCount):
            self.xStabilisers.append(scaleOperator(X, self.qubitCount, i + 1))

        self.zStabilisers = []
        for i in range(self.qubitCount):
            self.zStabilisers.append(scaleOperator(Z, self.qubitCount, i + 1))

        for i in range(len(self.grid)):
            for j in range(self.grid[i]):
                if j != len(self.grid[i]) - 1:
                    pass


    def __str__(self):
        returnStr = ""
        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                returnStr += "q "
            returnStr += "\n"
        return returnStr
    
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

    def errorChannel(self, _p):

        for i in range(len(self.qubitReferences)):
            if random.random() <= _p/3:
                self.qubitReferences[i].bitFlip()

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
                if self.qubitReferences[stabiliserNumber - 1].data == -1:
                    eigenvalue *= -1
            xSyndrome.append(0 if eigenvalue == 1 else 1)

        # print(f"Z Syndrome: {zSyndrome}")
        # print(f"X Syndrome: {xSyndrome}")

        return xSyndrome, zSyndrome

    # TODO: Make this scale with any size surface (find relation between syndrome vector and position on grid?)
    def errorCorrection(self, _xSyndrome, _zSyndrome):

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
        # Handle two bit-flip errors
        # else:
        #     stabiliserNums = []
        #     for index, value in enumerate(_zSyndrome):
        #         if value == 1: stabiliserNums.append(index)
        #     stabiliserCoordinates = []
        #     # Get coordinates of non-zero stabilisers on surface
        #     for value in stabiliserNums:
        #         stabiliserCoordinates.append(self.getSurfaceCoordinates("z", value + 1))
        #     stabiliserNeighbours = []
        #     # Get neighbouring data qubits of each stabiliser
        #     for row, col in stabiliserCoordinates:
        #         qubitNeighbours = []
        #         neighbours = self.getNeighbours(row, col)
        #         for neighbour in neighbours:
        #             if isinstance(neighbour, Qubit):
        #                 qubitNeighbours.append(neighbour)
        #         stabiliserNeighbours.append(qubitNeighbours)
        #     print(stabiliserNeighbours)

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

gridLayout = [
    ["q", "x", "q", "x", "q"],
    ["z", "q", "z", "q", "z"],
    ["q", "x", "q", "x", "q"],
    ["z", "q", "z", "q", "z"],
    ["q", "x", "q", "x", "q"]
]

surface = QubitSurface(gridLayout)
surface.qubitReferences[1].data = 1
surface.qubitReferences[6].data = 1
print([qubit.data for qubit in surface.qubitReferences])
xSyndrome, zSyndrome = surface.getErrorSyndromes()
surface.errorCorrection(xSyndrome, zSyndrome)
print([qubit.data for qubit in surface.qubitReferences])

initTime = time.time()
probabilityList = np.arange(0.005, 0.25, 0.005)
errorList = []
for i in range(len(probabilityList)):
    totalErrors = 0
    for j in range(CYCLES):
        surface = QubitSurface(gridLayout)
        initialQubitData = [qubit.data for qubit in surface.qubitReferences]
        surface.errorChannel(probabilityList[i])
        xSyndrome, zSyndrome = surface.getErrorSyndromes()
        surface.errorCorrection(xSyndrome, zSyndrome)
        finalQubitData = [qubit.data for qubit in surface.qubitReferences]
        if not np.all(initialQubitData == finalQubitData):
            for k in range(len(initialQubitData)):
                if initialQubitData[k] != finalQubitData[k]: totalErrors += 1
    errorList.append(totalErrors)
    print(f"Error probability: {probabilityList[i]}")
    print(f"Total errors after {CYCLES} cycles: {totalErrors}")
    print(f"Error rate: {totalErrors / (CYCLES)}")
    print("-----------------------------")

print(f"Time taken: {round(time.time() - initTime, 2)}s")

theoreticalModel = []
for i, prob in enumerate(probabilityList):
    theoreticalModel.append(1 - ((1-prob) ** 13) - 13 * prob * ((1 - prob) ** 12))

x = np.linspace(probabilityList[0], probabilityList[-1], 50)
theoreticalX = np.linspace(0, probabilityList[-1], 100)

z = np.polyfit(probabilityList, np.log(theoreticalModel), 3)
f = np.poly1d(z)
#y = f(x)
y = 1 - ((1-theoreticalX) ** 9) - 9 * theoreticalX * ((1 - theoreticalX) ** 8)

theoreticalPlot = plt.plot(theoreticalX, y, linestyle='dashed', color="black", label="Theoretical")

z = np.polyfit(probabilityList, [i / (CYCLES) for i in errorList], 3)
f = np.poly1d(z)
y = f(x)

plt.yscale("log")
dataPlot = plt.plot(probabilityList, [i / (CYCLES) for i in errorList], 'x', color="blue") 
dataCurvePlot = plt.plot(x, y, color="blue", label="Simulation")

plt.plot()

plt.show()
